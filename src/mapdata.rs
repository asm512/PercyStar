use crate::model3d::{extract_rage9_model_name, has_rage9_meshlets, read_virtual_from_file};
use crate::psin::joaat;
use flate2::{Decompress, FlushDecompress};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const ENTITY_STRUCT_HASH: u32 = 0xce50_1483;
pub const MAPDATA_STRUCT_HASH: u32 = 0xd359_3fa6;
pub const ARCHETYPE_DEF_HASH: u32 = 0x82d6_fc83;

#[derive(Clone, Copy, Debug)]
pub struct EntityPlacement {
    pub archetype_hash: u32,
    pub pos: [f32; 3],
    pub rot: [f32; 4], // world-space quaternion (x, y, z, w)
    pub scale_xy: f32,
    pub scale_z: f32,
}

#[derive(Default)]
pub struct WorldIndex {
    pub map_count: usize,
    pub models: HashMap<u32, u64>,
    pub entities: Vec<EntityPlacement>,
}

#[inline]
fn is_zlib(b: &[u8]) -> bool {
    b.len() >= 2 && b[0] == 0x78 && matches!(b[1], 0x01 | 0x5e | 0x9c | 0xda)
}
#[inline]
fn ru16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
fn ru32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn ru64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3], b[o + 4], b[o + 5], b[o + 6], b[o + 7]])
}
#[inline]
fn rf32(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn vptr(p: u64) -> Option<usize> {
    ((p >> 28) == 5).then_some((p & 0x0FFF_FFFF) as usize)
}

// decode 0DRPz (0x7a) map placement resource into world-space EntityPlacements
// layout credited to @holdmysocks (SixNG meta7a.rs)
pub fn parse_0drpz_map_entities(sys: &[u8]) -> Option<Vec<EntityPlacement>> {
    if sys.len() < 0x50 || sys.get(0x10..0x15) != Some(b"0DRPz") {
        return None;
    }
    let nbl = ru16(sys, 0x4C) as usize;
    let it = vptr(ru64(sys, 0x38))?;
    if it + nbl * 16 > sys.len() {
        return None;
    }

    let mut out = Vec::new();
    let mut has_map_root = false;
    for i in 0..nbl {
        let o = it + i * 16;
        let hash = ru32(sys, o);
        let sz = ru32(sys, o + 4) as usize;
        if hash == MAPDATA_STRUCT_HASH {
            has_map_root = true;
        }
        if hash != ENTITY_STRUCT_HASH || sz < 0x90 {
            continue;
        }
        let Some(base) = vptr(ru64(sys, o + 8)) else { continue };
        let end = base.saturating_add(sz).min(sys.len());
        let mut cur = base;
        while cur + 0x90 <= end {
            let arch = ru32(sys, cur + 0x44);
            let pos = [rf32(sys, cur + 0x38), rf32(sys, cur + 0x3C), rf32(sys, cur + 0x40)];
            // stored quaternion is inverse rotation; conjugate gives world rotation
            let rot = [
                -rf32(sys, cur + 0x10),
                -rf32(sys, cur + 0x14),
                -rf32(sys, cur + 0x18),
                rf32(sys, cur + 0x1C),
            ];
            let sxy = rf32(sys, cur + 0x54);
            let sz_s = rf32(sys, cur + 0x58);
            // an all zero rotation is an empty slot and would normalize to nan downstream
            let rot_len2: f32 = rot.iter().map(|v| v * v).sum();
            if pos.iter().all(|v| v.is_finite() && v.abs() < 30_000.0) && rot_len2.is_finite() && rot_len2 > 1e-6 {
                out.push(EntityPlacement {
                    archetype_hash: arch,
                    pos,
                    rot,
                    scale_xy: if sxy.is_finite() && sxy > 0.01 && sxy < 100.0 { sxy } else { 1.0 },
                    scale_z: if sz_s.is_finite() && sz_s > 0.01 && sz_s < 100.0 { sz_s } else { 1.0 },
                });
            }
            cur += 0x90;
        }
    }
    if has_map_root || !out.is_empty() {
        Some(out)
    } else {
        None
    }
}

pub fn parse_0drpz_archetype_aliases(sys: &[u8]) -> Vec<(u32, u32)> {
    if sys.len() < 0x50 || sys.get(0x10..0x15) != Some(b"0DRPz") {
        return Vec::new();
    }
    let nbl = ru16(sys, 0x4C) as usize;
    let Some(it) = vptr(ru64(sys, 0x38)) else { return Vec::new() };
    if it + nbl * 16 > sys.len() {
        return Vec::new();
    }
    let mut aliases = Vec::new();
    for i in 0..nbl {
        let o = it + i * 16;
        let hash = ru32(sys, o);
        let sz = ru32(sys, o + 4) as usize;
        if hash != ARCHETYPE_DEF_HASH || sz < 0xB8 {
            continue;
        }
        let Some(base) = vptr(ru64(sys, o + 8)) else { continue };
        let end = base.saturating_add(sz).min(sys.len());
        let mut cur = base;
        while cur + 0xB8 <= end {
            let arch = ru32(sys, cur + 0x74);
            let draw = ru32(sys, cur + 0x7C);
            let asset = ru32(sys, cur + 0x88);
            if arch != 0 && draw != 0 && draw != arch {
                aliases.push((arch, draw));
            }
            if arch != 0 && asset != 0 && asset != arch {
                aliases.push((arch, asset));
            }
            cur += 0xB8;
        }
    }
    aliases
}

fn merge_scan_cache_models(ix: &mut WorldIndex) {
    let Some(base) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) else { return };
    let tsv = base.join("PercyStar").join("scan_cache_v9.tsv");
    let Ok(f) = File::open(tsv) else { return };
    use std::io::BufRead;
    let r = BufReader::new(f);
    for line in r.lines().map_while(Result::ok) {
        if !line.contains("\tlevels.rpf\t") {
            continue;
        }
        let mut cols = line.split('\t');
        let (_k, _p, _rpf, Some(name), Some(kind), Some(off_s)) = (
            cols.next(), cols.next(), cols.next(), cols.next(), cols.next(), cols.next()
        ) else { continue };
        if kind != "MODEL" && kind != "VEH" {
            continue;
        }
        let Ok(off) = off_s.parse::<u64>() else { continue };
        let stem = name.split(".#").next().unwrap_or(name);
        ix.models.entry(joaat(stem)).or_insert(off);
    }
}

fn cache_path() -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    Some(base.join("PercyStar").join("world_index_v2.bin"))
}

fn save_world_index(path: &Path, rpf_len: u64, ix: &WorldIndex) {
    if let Some(par) = path.parent() {
        let _ = fs::create_dir_all(par);
    }
    let Ok(f) = File::create(path) else { return };
    let mut w = BufWriter::new(f);
    let _ = w.write_all(b"PSW2");
    let _ = w.write_all(&rpf_len.to_le_bytes());
    let _ = w.write_all(&(ix.map_count as u64).to_le_bytes());
    let _ = w.write_all(&(ix.models.len() as u64).to_le_bytes());
    let _ = w.write_all(&(ix.entities.len() as u64).to_le_bytes());
    for (&h, &off) in &ix.models {
        let _ = w.write_all(&h.to_le_bytes());
        let _ = w.write_all(&off.to_le_bytes());
    }
    for e in &ix.entities {
        let _ = w.write_all(&e.archetype_hash.to_le_bytes());
        for v in e.pos { let _ = w.write_all(&v.to_le_bytes()); }
        for v in e.rot { let _ = w.write_all(&v.to_le_bytes()); }
        let _ = w.write_all(&e.scale_xy.to_le_bytes());
        let _ = w.write_all(&e.scale_z.to_le_bytes());
    }
    let _ = w.flush();
}

fn load_world_index(path: &Path, rpf_len: u64) -> Option<WorldIndex> {
    let mut r = BufReader::new(File::open(path).ok()?);
    let mut hdr = [0u8; 36];
    r.read_exact(&mut hdr).ok()?;
    if &hdr[..4] != b"PSW2" || ru64(&hdr, 4) != rpf_len {
        return None;
    }
    let map_count = ru64(&hdr, 12) as usize;
    let n_models = ru64(&hdr, 20) as usize;
    let n_ents = ru64(&hdr, 28) as usize;
    // counts are cache file values they must fit in what is left of the file
    let body = fs::metadata(path).ok()?.len().saturating_sub(36);
    if (n_models as u64).saturating_mul(12).saturating_add((n_ents as u64).saturating_mul(40)) > body {
        return None;
    }
    let mut models = HashMap::with_capacity(n_models);
    let mut mbuf = [0u8; 12];
    for _ in 0..n_models {
        r.read_exact(&mut mbuf).ok()?;
        models.insert(ru32(&mbuf, 0), ru64(&mbuf, 4));
    }
    let mut entities = Vec::with_capacity(n_ents);
    let mut ebuf = [0u8; 40];
    for _ in 0..n_ents {
        r.read_exact(&mut ebuf).ok()?;
        entities.push(EntityPlacement {
            archetype_hash: ru32(&ebuf, 0),
            pos: [rf32(&ebuf, 4), rf32(&ebuf, 8), rf32(&ebuf, 12)],
            rot: [rf32(&ebuf, 16), rf32(&ebuf, 20), rf32(&ebuf, 24), rf32(&ebuf, 28)],
            scale_xy: rf32(&ebuf, 32),
            scale_z: rf32(&ebuf, 36),
        });
    }
    Some(WorldIndex { map_count, models, entities })
}

pub fn build_or_load_world_index(levels_rpf: &Path) -> WorldIndex {
    let Ok(f) = File::open(levels_rpf) else { return WorldIndex::default() };
    let sz = f.metadata().map(|m| m.len()).unwrap_or(0);
    if let Some(cp) = cache_path() {
        if let Some(mut ix) = load_world_index(&cp, sz) {
            merge_scan_cache_models(&mut ix);
            return ix;
        }
    }

    let chunk_sz = 64 * 1024 * 1024u64;
    let next_off = std::sync::atomic::AtomicU64::new(0);
    let done_bytes = std::sync::atomic::AtomicU64::new(0);
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).clamp(2, 12);

    let mut ix = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        for tid in 0..n_threads {
            let next_off = &next_off;
            let done_bytes = &done_bytes;
            handles.push(s.spawn(move || {
                let mut local = WorldIndex::default();
                let mut local_aliases: Vec<(u32, u32)> = Vec::new();
                let Ok(mut f_scan) = File::open(levels_rpf) else { return (local, local_aliases) };
                let Ok(mut f_res) = File::open(levels_rpf) else { return (local, local_aliases) };
                let mut buf = vec![0u8; chunk_sz as usize];
                let mut dec = Decompress::new(true);
                let mut head = [0u8; 64];
                let mut out = vec![0u8; 32768];
                loop {
                    let off = next_off.fetch_add(chunk_sz, std::sync::atomic::Ordering::Relaxed);
                    if off >= sz {
                        break;
                    }
                    if f_scan.seek(SeekFrom::Start(off)).is_err() {
                        break;
                    }
                    let want = (sz - off).min(chunk_sz) as usize;
                    let mut read_n = 0usize;
                    while read_n < want {
                        match f_scan.read(&mut buf[read_n..want]) {
                            Ok(0) => break,
                            Ok(n) => read_n += n,
                            Err(_) => break,
                        }
                    }
                    let mut pos = 0usize;
                    while pos + 0x20 <= read_n {
                        let p = pos + 0x10;
                        if is_zlib(&buf[p..]) {
                            dec.reset(true);
                            let _ = dec.decompress(&buf[p..(p + 256).min(read_n)], &mut head, FlushDecompress::None);
                            if dec.total_out() >= 0x20 {
                                let vft = ru64(&head, 0);
                                let bmap = ru64(&head, 8);
                                if vft == 0 && (0x5000_0000..0x5800_0000).contains(&bmap) {
                                    let abs_off = off + p as u64;
                                    if head.get(0x10..0x15) == Some(b"0DRPz") {
                                        if let Ok((sys, _, _)) = read_virtual_from_file(&mut f_res, abs_off) {
                                            if let Some(ents) = parse_0drpz_map_entities(&sys) {
                                                local.map_count += 1;
                                                local.entities.extend(ents);
                                            }
                                            for (arch, draw) in parse_0drpz_archetype_aliases(&sys) {
                                                local_aliases.push((arch, draw));
                                            }
                                        }
                                    } else {
                                        dec.reset(true);
                                        let _ = dec.decompress(&buf[p..(p + 32768).min(read_n)], &mut out, FlushDecompress::None);
                                        let olen = dec.total_out() as usize;
                                        if olen >= 0x80 && has_rage9_meshlets(&out[..olen]) {
                                            if let Some((nm, ext)) = extract_rage9_model_name(&out[..olen]) {
                                                let stem = nm.strip_suffix(ext).unwrap_or(&nm);
                                                local.models.entry(joaat(stem)).or_insert(abs_off);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        pos += 0x1000;
                    }
                    let done = done_bytes.fetch_add(read_n as u64, std::sync::atomic::Ordering::Relaxed) + read_n as u64;
                    if tid == 0 {
                        eprintln!("[world-index] {:.1} / {:.1} GB...", done as f64 / 1e9, sz as f64 / 1e9);
                    }
                }
                (local, local_aliases)
            }));
        }
        let mut merged = WorldIndex::default();
        let mut all_aliases = Vec::new();
        for h in handles {
            if let Ok((local, aliases)) = h.join() {
                merged.map_count += local.map_count;
                merged.entities.extend(local.entities);
                for (k, v) in local.models {
                    merged.models.entry(k).or_insert(v);
                }
                all_aliases.extend(aliases);
            }
        }
        for (arch, draw) in all_aliases {
            if let Some(&off) = merged.models.get(&draw) {
                merged.models.entry(arch).or_insert(off);
            }
        }
        merged
    });

    merge_scan_cache_models(&mut ix);
    if let Some(cp) = cache_path() {
        save_world_index(&cp, sz, &ix);
    }
    ix
}

pub fn probe_maps_cli(rpf_path: &Path) {
    let ix = build_or_load_world_index(rpf_path);
    println!("WorldIndex: {} maps, {} models, {} entity placements", ix.map_count, ix.models.len(), ix.entities.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_synthetic_0drpz_map_entities() {
        let mut sys = vec![0u8; 0x180];
        sys[0x10..0x15].copy_from_slice(b"0DRPz");
        sys[0x4C..0x4E].copy_from_slice(&2u16.to_le_bytes()); // 2 blocks
        sys[0x38..0x40].copy_from_slice(&(0x5000_0080u64).to_le_bytes());
        // block 0: MAPDATA
        sys[0x80..0x84].copy_from_slice(&MAPDATA_STRUCT_HASH.to_le_bytes());
        // block 1: ENTITY (1 * 0x90) at 0x5000_00E0
        sys[0x90..0x94].copy_from_slice(&ENTITY_STRUCT_HASH.to_le_bytes());
        sys[0x94..0x98].copy_from_slice(&0x90u32.to_le_bytes());
        sys[0x98..0xA0].copy_from_slice(&(0x5000_00E0u64).to_le_bytes());
        let e = 0xE0;
        sys[e + 0x44..e + 0x48].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        sys[e + 0x38..e + 0x3C].copy_from_slice(&(-2351.0f32).to_le_bytes());
        sys[e + 0x3C..e + 0x40].copy_from_slice(&(-5533.0f32).to_le_bytes());
        sys[e + 0x40..e + 0x44].copy_from_slice(&8.0f32.to_le_bytes());
        sys[e + 0x1C..e + 0x20].copy_from_slice(&1.0f32.to_le_bytes());
        sys[e + 0x54..e + 0x58].copy_from_slice(&1.5f32.to_le_bytes());
        sys[e + 0x58..e + 0x5C].copy_from_slice(&2.0f32.to_le_bytes());

        let ents = parse_0drpz_map_entities(&sys).unwrap();
        assert_eq!(ents.len(), 1);
        assert_eq!(ents[0].archetype_hash, 0x1234_5678);
        assert_eq!(ents[0].pos, [-2351.0, -5533.0, 8.0]);
        assert_eq!(ents[0].scale_xy, 1.5);
        assert_eq!(ents[0].scale_z, 2.0);
    }
}
