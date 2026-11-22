use std::fmt::Write as _;

fn rd32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

fn rd64(d: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3], d[o + 4], d[o + 5], d[o + 6], d[o + 7]])
}

fn vptr_off(ptr: u64) -> Option<usize> {
    if (0x5000_0000..0x5800_0000).contains(&ptr) {
        Some((ptr - 0x5000_0000) as usize)
    } else {
        None
    }
}

// extract primary duration in seconds, track count, and clip hash seed from crClipDictionary
pub fn extract_anim_summary(data: &[u8]) -> (f32, usize, u32) {
    let mut dur = 2.0f32;
    // the seed fields below sit at 0x20 and 0x28
    if data.len() < 0x30 {
        return (dur, 1, 0);
    }
    for o in (0x60..data.len().saturating_sub(0x38)).step_by(4) {
        let d = f32::from_bits(rd32(data, o));
        if d.is_finite() && (0.05..=3600.0).contains(&d) {
            let p0 = rd32(data, o + 4);
            let p1 = rd32(data, o + 8);
            if p0 == p1 && (0x5000_0000..0x5800_0000).contains(&p0) {
                dur = d;
                break;
            }
        }
    }
    let tracks = data.windows(4).filter(|w| w[0] == 0x09 && w[1] == 0x00).count().max(1);
    let mut seed = rd32(data, 0x20);
    if let Some(hoff) = vptr_off(rd64(data, 0x28)) {
        if hoff + 8 <= data.len() {
            seed ^= rd32(data, hoff).rotate_left(7) ^ rd32(data, hoff + 4);
        }
    }
    (dur, tracks, seed)
}

fn push_limb(
    verts: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    tris: &mut Vec<[u32; 3]>,
    edges: &mut Vec<(u32, u32)>,
    a: [f32; 3],
    b: [f32; 3],
    r: f32,
) {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    let len = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-4);
    let ux = dx / len;
    let uy = dy / len;
    let uz = dz / len;
    let ref_v = if uy.abs() < 0.9 { [0.0f32, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let px = uy * ref_v[2] - uz * ref_v[1];
    let py = uz * ref_v[0] - ux * ref_v[2];
    let pz = ux * ref_v[1] - uy * ref_v[0];
    let plen = (px * px + py * py + pz * pz).sqrt().max(1e-4);
    let s1 = [px / plen * r, py / plen * r, pz / plen * r];
    let s2 = [
        (uy * s1[2] - uz * s1[1]),
        (uz * s1[0] - ux * s1[2]),
        (ux * s1[1] - uy * s1[0]),
    ];
    let base = verts.len() as u32;
    let ring = |p: [f32; 3]| -> [[f32; 3]; 4] {
        [
            [p[0] + s1[0], p[1] + s1[1], p[2] + s1[2]],
            [p[0] + s2[0], p[1] + s2[1], p[2] + s2[2]],
            [p[0] - s1[0], p[1] - s1[1], p[2] - s1[2]],
            [p[0] - s2[0], p[1] - s2[1], p[2] - s2[2]],
        ]
    };
    for v in ring(a).into_iter().chain(ring(b)) {
        verts.push(v);
        uvs.push([0.5, 0.5]);
    }
    for k in 0..4u32 {
        let kn = (k + 1) % 4;
        tris.push([base + k, base + kn, base + 4 + kn]);
        tris.push([base + k, base + 4 + kn, base + 4 + k]);
        edges.push((base + k, base + 4 + k));
        edges.push((base + k, base + kn));
    }
}

// build z-up articulated 3d stickman dummy posed at time_sec (-y faces camera)
pub fn build_stickman_mesh(
    name: &str,
    time_sec: f32,
    duration_sec: f32,
    track_count: usize,
    seed: u32,
) -> crate::model3d::Mesh3D {
    let freq = 1.0 + ((seed & 0xF) as f32) * 0.15;
    let phase = (time_sec / duration_sec.max(0.1)) * std::f32::consts::TAU * freq;
    let sw = phase.sin();
    let cw = phase.cos();
    let mode = seed % 4;
    let stride = match mode {
        0 => 0.24,  // walk/stride
        1 => 0.04,  // idle/subtle shift
        2 => 0.10,  // gesture/reach
        _ => 0.16,  // crouch/action
    };
    let crouch = if mode == 3 { 0.18 } else { 0.0 };
    let arm_raise = match mode {
        2 => 0.35 + 0.18 * sw,
        3 => 0.20,
        _ => 0.0,
    };

    // z is up, -y is forward toward camera
    let pelvis = [0.0, 0.0, 0.95 - crouch + 0.03 * (phase * 2.0).sin().abs()];
    let chest = [0.03 * sw, -0.02 * cw - crouch * 0.3, pelvis[2] + 0.45];
    let neck = [chest[0], chest[1], chest[2] + 0.14];
    let head_top = [neck[0] + 0.02 * cw, neck[1] - 0.02, neck[2] + 0.22];

    let l_sh = [chest[0] - 0.22, chest[1], chest[2] - 0.03];
    let r_sh = [chest[0] + 0.22, chest[1], chest[2] - 0.03];
    let l_el = [l_sh[0] - 0.08, l_sh[1] - 0.20 * sw - arm_raise * 0.4, l_sh[2] - 0.26 + arm_raise * 0.5];
    let r_el = [r_sh[0] + 0.08, r_sh[1] + 0.20 * sw - arm_raise * 0.4, r_sh[2] - 0.26 + arm_raise * 0.5];
    let l_ha = [l_el[0] - 0.03, l_el[1] - 0.22 * sw - 0.12 - arm_raise * 0.5, l_el[2] - 0.20 + arm_raise * 0.7];
    let r_ha = [r_el[0] + 0.03, r_el[1] + 0.22 * sw - 0.12 - arm_raise * 0.5, r_el[2] - 0.20 + arm_raise * 0.7];

    let l_hi = [pelvis[0] - 0.11, pelvis[1], pelvis[2] - 0.03];
    let r_hi = [pelvis[0] + 0.11, pelvis[1], pelvis[2] - 0.03];
    let l_kn = [l_hi[0], l_hi[1] + stride * sw - crouch * 0.4, l_hi[2] - 0.42 + crouch * 0.3];
    let r_kn = [r_hi[0], r_hi[1] - stride * sw - crouch * 0.4, r_hi[2] - 0.42 + crouch * 0.3];
    let l_ft = [l_kn[0], l_kn[1] + stride * 0.8 * sw + 0.05 * cw, (l_kn[2] - 0.44).max(0.02)];
    let r_ft = [r_kn[0], r_kn[1] - stride * 0.8 * sw - 0.05 * cw, (r_kn[2] - 0.44).max(0.02)];

    let mut verts = Vec::with_capacity(160);
    let mut uvs = Vec::with_capacity(160);
    let mut tris = Vec::with_capacity(160);
    let mut edges = Vec::with_capacity(160);

    for (a, b, r) in [
        (pelvis, chest, 0.055),
        (chest, neck, 0.045),
        (neck, head_top, 0.095),
        (l_sh, r_sh, 0.038),
        (l_sh, l_el, 0.032),
        (l_el, l_ha, 0.026),
        (r_sh, r_el, 0.032),
        (r_el, r_ha, 0.026),
        (l_hi, r_hi, 0.042),
        (l_hi, l_kn, 0.038),
        (l_kn, l_ft, 0.030),
        (r_hi, r_kn, 0.038),
        (r_kn, r_ft, 0.030),
    ] {
        push_limb(&mut verts, &mut uvs, &mut tris, &mut edges, a, b, r);
    }

    let tri_geom = vec![0u16; tris.len()];
    let mut mesh = crate::model3d::Mesh3D::from_parts(
        name.to_string(),
        verts,
        uvs,
        tris,
        tri_geom,
        edges,
        None,
        format!("Stickman Rig ({track_count} tracks)"),
        1,
        1,
        format!("{duration_sec:.2}s"),
    );
    mesh.shade_mode = crate::model3d::ShadeMode::SolidWire;
    mesh
}

// detect rage 9 crClipDictionary header in decompressed rsc9 page
pub fn probe_anim_dict(data: &[u8]) -> Option<(String, u32)> {
    if data.len() < 0x50 {
        return None;
    }
    if rd64(data, 0) != 0 || vptr_off(rd64(data, 8)).is_none() {
        return None;
    }
    if &data[0x10..0x15] != b"0DRPz" {
        return None;
    }
    let count = rd32(data, 0x20);
    if count == 0 || count > 4096 {
        return None;
    }
    let p_hashes = vptr_off(rd64(data, 0x28))?;
    let p_anims = vptr_off(rd64(data, 0x38));
    if p_hashes == 0 && p_anims.is_none() {
        return None;
    }
    let first_hash = if p_hashes + 4 <= data.len() { rd32(data, p_hashes) } else { 0 };
    Some((format!("clipdict_{first_hash:08x}_{count}clips.ycd"), count))
}

// decode rage 9 crClipDictionary header hash table and anim entries
pub fn parse_anim_dict(data: &[u8]) -> String {
    if data.len() < 0x50 || &data[0x10..0x15] != b"0DRPz" {
        return String::from("Not a valid RAGE 9 crClipDictionary (0DRPz)");
    }
    let bmap = rd64(data, 8);
    let count = rd32(data, 0x20);
    let p_hashes = rd64(data, 0x28);
    let p_clips = rd64(data, 0x30);
    let p_anims = rd64(data, 0x38);
    let p_skel = rd64(data, 0x40);

    let mut out = String::with_capacity(1024);
    let _ = writeln!(
        out,
        "=== RAGE 9 crClipDictionary (ANIM) ===\nSize: {} B | BlockMap: 0x{:08x} | Clips: {}",
        data.len(),
        bmap,
        count
    );
    let _ = writeln!(
        out,
        "Tables: hashes=0x{:08x} clips=0x{:08x} anims=0x{:08x} skel=0x{:08x}\n",
        p_hashes, p_clips, p_anims, p_skel
    );

    // scan for crAnimation duration floats (followed by 0x50xxxxxx block pointers)
    let mut durations = Vec::new();
    for o in (0x60..data.len().saturating_sub(0x38)).step_by(4) {
        let dur = f32::from_bits(rd32(data, o));
        if dur.is_finite() && (0.05..=3600.0).contains(&dur) {
            let p0 = rd32(data, o + 4);
            let p1 = rd32(data, o + 8);
            if p0 == p1 && (0x5000_0000..0x5800_0000).contains(&p0) {
                durations.push((o, dur));
                if durations.len() >= 16 {
                    break;
                }
            }
        }
    }
    if !durations.is_empty() {
        let _ = writeln!(out, "--- Decoded crAnimation Timelines ---");
        for (o, dur) in &durations {
            let frames = (dur * 30.0).round() as u32;
            let _ = writeln!(out, "  @0x{o:05x} duration={dur:.3}s (~{frames} frames @ 30fps)");
        }
        out.push('\n');
    }

    if let Some(hoff) = vptr_off(p_hashes) {
        let _ = writeln!(out, "--- Clip Hash Entries ---");
        for idx in 0..(count as usize).min(64) {
            let e = hoff + idx * 32;
            if e + 32 > data.len() {
                break;
            }
            let chash = rd32(data, e);
            let flags = rd32(data, e + 8);
            let cptr = rd64(data, e + 16);
            let _ = writeln!(out, "  [{idx:02}] hash=0x{chash:08x} flags=0x{flags:04x} ptr=0x{cptr:08x}");
        }
    }

    if let Some(aoff) = vptr_off(p_anims) {
        let _ = writeln!(out, "\n--- Animation Blocks ---");
        for idx in 0..(count as usize).min(64) {
            let e = aoff + 16 + idx * 16;
            if e + 16 > data.len() {
                break;
            }
            let ahash = rd32(data, e);
            let asize = rd32(data, e + 4);
            let aptr = rd64(data, e + 8);
            let tracks = vptr_off(aptr)
                .filter(|&o| o + 32 <= data.len())
                .map(|o| {
                    let end = (o + asize as usize).min(data.len());
                    data[o..end].windows(4).filter(|w| w[0] == 0x09 && w[1] == 0x00).count()
                })
                .unwrap_or(0);
            let _ = writeln!(out, "  [{idx:02}] anim=0x{ahash:08x} block_sz=0x{asize:04x} ptr=0x{aptr:08x} tracks~={tracks}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_of_a_short_buffer_does_not_panic() {
        for len in [0usize, 4, 0x20, 0x24, 0x2f, 0x30, 0x60, 0x98, 0x200] {
            let (dur, tracks, _) = extract_anim_summary(&vec![0u8; len]);
            assert!(dur > 0.0 && tracks >= 1);
        }
    }

    #[test]
    fn probes_and_parses_synthetic_clip_dict() {
        let mut buf = vec![0u8; 0x100];
        buf[8..16].copy_from_slice(&0x5000_00c0u64.to_le_bytes());
        buf[0x10..0x15].copy_from_slice(b"0DRPz");
        buf[0x20..0x24].copy_from_slice(&2u32.to_le_bytes());
        buf[0x28..0x30].copy_from_slice(&0x5000_0060u64.to_le_bytes());
        buf[0x38..0x40].copy_from_slice(&0x5000_00a0u64.to_le_bytes());
        buf[0x60..0x64].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        buf[0x8c..0x90].copy_from_slice(&48.133335f32.to_le_bytes());
        buf[0x90..0x94].copy_from_slice(&0x5015_02f9u32.to_le_bytes());
        buf[0x94..0x98].copy_from_slice(&0x5015_02f9u32.to_le_bytes());
        let (name, cnt) = probe_anim_dict(&buf).expect("should detect clip dict");
        assert_eq!(cnt, 2);
        assert!(name.contains("deadbeef"));
        let txt = parse_anim_dict(&buf);
        assert!(txt.contains("Clips: 2"));
        assert!(txt.contains("0xdeadbeef"));
        assert!(txt.contains("duration=48.133s (~1444 frames @ 30fps)"));
        let (dur, trk, seed) = extract_anim_summary(&buf);
        assert!((dur - 48.133335).abs() < 0.01);
        assert!(trk >= 1);
        let m = build_stickman_mesh("test_clip", 0.5, dur, trk, seed);
        assert!(!m.verts.is_empty());
        assert!(!m.tris.is_empty());
    }
}
