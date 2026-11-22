// `PercyStar --bench <game dir> [runs]`: times the real hot paths on warm files, best of N runs
use super::*;

fn time<T>(label: &str, runs: usize, mut f: impl FnMut() -> T) -> T {
    let mut best = f64::MAX;
    let mut out = None;
    for _ in 0..runs.max(1) {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        out = Some(r);
    }
    println!("{label:<34} {best:>10.1} ms");
    out.unwrap()
}

fn digest(e: &[RpfEntry]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for e in e {
        (&e.rpf_name, &e.name, e.kind, e.offset, e.comp_size, e.raw_size, e.tracks, &e.chunks).hash(&mut h);
    }
    h.finish()
}

fn px_digest<'a>(t: impl Iterator<Item = &'a DecodedTexture>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for t in t {
        (&t.name, t.w, t.h, t.dxgi).hash(&mut h);
        for c in &t.pixels {
            c.to_array().hash(&mut h);
        }
    }
    h.finish()
}

pub fn run(base: &Path, runs: usize, only: &str) {
    let want = |k: &str| only.is_empty() || only.split(',').any(|o| k.contains(o));
    println!("PercyStar {APP_VERSION} bench, best of {runs}, {}", base.display());

    if let Some(cache) = cache_file_path().filter(|p| want("cache") && p.exists()) {
        let map = time("cache: load index", runs, || load_disk_cache_from(&cache));
        let mut keys: Vec<_> = map.iter().map(|(k, v)| (k.clone(), digest(v))).collect();
        keys.sort();
        println!("  ({} archives, digest {:016x})", keys.len(), { use std::hash::{Hash, Hasher}; let mut h = std::collections::hash_map::DefaultHasher::new(); keys.hash(&mut h); h.finish() });
        let tmp = std::env::temp_dir().join("percystar_bench_cache.tsv");
        time("cache: save index", runs, || save_disk_cache_to(&map, &tmp));
        let back = load_disk_cache_from(&tmp);
        let same = back.len() == map.len() && map.iter().all(|(k, v)| back.get(k).is_some_and(|b| digest(b) == digest(v)));
        println!("  (save then load gives the same index: {same})");
        let _ = fs::remove_file(&tmp);
        let entries: Vec<RpfEntry> = time("cache: fill entry list", runs, || keys.iter().flat_map(|(k, _)| map[k].iter().cloned()).collect());
        let hay = time("search: build index", runs, || build_hay(&entries));
        println!("  (index digest {:016x})", { use std::hash::{Hash, Hasher}; let mut h = std::collections::hash_map::DefaultHasher::new(); hay.hash(&mut h); h.finish() });
        for q in ["adder", "tex_ bc7 -mip", "zzzz_no_match"] {
            let hits = time(&format!("search: query '{q}'"), runs, || {
                let (inc, exc): (Vec<&str>, Vec<&str>) = q.split_whitespace().partition(|t| !t.starts_with('-'));
                let exc: Vec<&str> = exc.iter().map(|t| &t[1..]).collect();
                par_filter(hay.len(), |i| inc.iter().all(|t| hay[i].contains(t)) && !exc.iter().any(|t| hay[i].contains(t))).len()
            });
            println!("  ({hits} hits)");
        }
        println!("  ({} entries)", entries.len());
    }

    let mut scans = HashMap::new();
    for name in ["script.rpf", "common.rpf", "textures.rpf", "shaders.rpf", "data.rpf", "packs.rpf"] {
        let p = base.join(name);
        if want(name) && p.exists() {
            let (_, e) = time(&format!("scan: {name}"), runs, || scan_single_rpf(&p, None).unwrap_or_default());
            println!("  ({} entries, digest {:016x})", e.len(), digest(&e));
            scans.insert(name, e);
        }
    }
    let ttpf = base.join(r"data\ttdb\graphics\ttdb_graphics.ttpf");
    if want("ttpf") && ttpf.exists() {
        let (_, d) = time("scan: ttdb_graphics.ttpf", runs, || scan_ttpf_descriptors(&ttpf).unwrap());
        let ids: Vec<usize> = d.iter().map(|d| d.idx).take(60).collect();
        let px = time("decode: 60 TTPF textures", runs, || ids.iter().filter_map(|&i| decode_ttpf_texture(&ttpf, i).ok()).collect::<Vec<_>>());
        println!("  (pixel digest {:016x})", px_digest(px.iter()));
    }
    if let Some(e) = scans.get("textures.rpf").filter(|_| want("txd")) {
        let tex = base.join("textures.rpf");
        let offs: Vec<u64> = e.iter().filter(|e| e.kind == "TXD").map(|e| e.offset).take(40).collect();
        let px = time("decode: 40 TXD dictionaries", runs, || offs.iter().filter_map(|&o| decode_txd_from_rpf(&tex, o).ok()).collect::<Vec<_>>());
        println!("  (pixel digest {:016x})", px_digest(px.iter().flatten()));
    }
    if let Some(e) = scans.get("packs.rpf").filter(|_| want("model")) {
        let packs = base.join("packs.rpf");
        let models: Vec<&RpfEntry> = e.iter().filter(|e| e.kind == "MODEL" || e.kind == "VEH").take(25).collect();
        let n = time("decode: 25 packs.rpf models", runs, || {
            models.iter().filter_map(|m| decode_rage9_model_from_rpf(&packs, m.offset, &m.name).ok()).map(|m| (m.verts.len(), m.tris.len())).collect::<Vec<_>>()
        });
        println!("  ({} decoded, {:?})", n.len(), n.iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1)));
    }
    let levels = base.join("levels.rpf");
    if only.contains("veh") && levels.exists() {
        let e = time("scan: levels.rpf vehicle zone", runs, || scan_levels_vehicle_zone(&levels, None));
        println!("  ({} entries, digest {:016x})", e.len(), digest(&e));
    }
}
