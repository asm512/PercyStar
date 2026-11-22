use crate::psin::joaat;
use flate2::{Decompress, FlushDecompress, Status};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn rd32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

// detect rage 9 audio metadata rel container (dat18 dat20 dat25 dat26 dat41 dat61 dat86 dat166 dat581)
pub fn probe_dat(head: &[u8]) -> Option<u32> {
    if head.len() < 16 {
        return None;
    }
    let ver = rd32(head, 0);
    let data_sz = rd32(head, 4);
    let ts = rd32(head, 8);
    if matches!(ver, 16 | 18 | 19 | 20 | 25 | 26 | 29 | 41 | 61 | 86 | 151 | 166 | 581)
        && (4..=64 * 1024 * 1024).contains(&data_sz)
        && ts != 0
        && ts != 0xFFFF_FFFF
    {
        Some(ver)
    } else {
        None
    }
}

#[derive(Clone, Debug)]
pub struct AudioBankMeta {
    pub bank_hash: u32,
    pub exact_size: u64,
    pub aligned_size: u64,
    pub pack_name: String,
    pub used_by: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedAwcMeta {
    pub name: String,
    pub exact_size: u64,
    pub bank_hash: u32,
    pub pack_name: String,
    pub used_by: Vec<String>,
}

pub struct AudioRpfCatalog {
    pub banks: Vec<AudioBankMeta>,
    pub metadata_chunks: Vec<(String, &'static str, u64, u64, Vec<(u64, u64)>)>,
}

fn is_zlib_hdr(d: &[u8]) -> bool {
    if d.len() < 2 {
        return false;
    }
    let b0 = d[0] as u16;
    let b1 = d[1] as u16;
    b0 == 0x78 && matches!(b1, 0x01 | 0x5E | 0x9C | 0xDA) && ((b0 << 8) | b1).is_multiple_of(31)
}

fn known_pack_fallback(pack_hash: u32) -> Option<&'static str> {
    match pack_hash {
        // Large 2840-bank streaming pack (Music, Radio, Score, Event streams)
        0x2df8_3b87 => Some("Streams_Music_Radio"),
        // Verified via JOAAT hash_dict
        0x23d1_d81c => Some("Speech.Ped"),
        0x8856_f65a => Some("Props"),
        0x4223_1ebd => Some("Speech.Player"),
        0xe7a8_cdf1 => Some("Script"),
        0x75fa_4d0c => Some("Animals"),
        0x3bda_a15c => Some("Ambience.WaveLoaded"),
        0x21b8_32cc => Some("Streams"),
        0x146c_bf51 => Some("Ambience.Streams"),
        0xaaf7_518a => Some("Resident.Vehicles"),
        0x803e_392c => Some("Weapons"),
        0xe8f6_8d48 => Some("Radio.DjSpeech"),
        0x76c4_9ce5 => Some("Resident.Water"),
        0x78dc_5e8b => Some("Resident.Props"),
        0x87c4_912b => Some("Doors"),
        0x0acd_9bbf => Some("Resident.Peds"),
        0x513b_c04f => Some("Resident.Collisions"),
        0xeb90_ab5f => Some("Resident.Cloth"),
        0x6329_09d2 => Some("Resident.Surfaces"),
        0x553a_b4d2 => Some("Walla"),
        0x1951_f94c => Some("Cloth"),
        0x677a_47fa => Some("Resident.Bullets"),
        0xa93b_d90c => Some("Vehicles.Trains"),
        0x49b2_0d57 => Some("Resident.Frontend"),
        0x3c90_e44f => Some("Resident.Doors"),
        0xc178_1034 => Some("Resident.Weather"),
        0x32ff_02d1 => Some("Resident.Ambience"),
        0xd346_0b2e => Some("Speech.Silent"),
        0x75b7_478e => Some("Vehicles.Boats"),
        0x2314_98de => Some("Resident"),
        // Verified via WaveSlot bank_hash links in DAT26
        0xc268_d7dc => Some("PoliceScanner"),
        0x32b4_cd63 => Some("Vehicles.Damage"),
        0xc6fe_b1e3 => Some("Vehicles.Aircraft_Damage"),
        0x496c_4b20 => Some("Resident.Glass"),
        0xb118_0a9b => Some("Resident.Explosions"),
        0xdcd1_626f => Some("Resident.Animals"),
        0x7e03_aacc => Some("Walla.Interior"),
        // Verified via DAT166 / DAT581 / speech2_common references
        0x76f9_ac27 => Some("Animals.Wildlife"),
        0x2801_af67 => Some("Ambience.Nature"),
        0xae4f_1d61 => Some("Ambience.Industrial"),
        0x7ebe_2855 => Some("Ambience.Distant"),
        0xe5d9_ea62 => Some("Speech.Common"),
        _ => None,
    }
}

/// Parses the 19.5 MB zlib-compressed RAGE 9 audio metadata prefix at the beginning of `audio.rpf`,
/// extracting the `BankSizes` table, resolving `pack_hash` categories and `WaveSlot` / `DAT166` sound labels.
pub fn parse_audio_rpf_catalog(audio_rpf: &Path, first_awc_off: u64) -> Option<AudioRpfCatalog> {
    let limit = (first_awc_off as usize).clamp(64 * 1024, 24 * 1024 * 1024);
    let mut f = File::open(audio_rpf).ok()?;
    let mut data = vec![0u8; limit];
    let n = f.read(&mut data).ok()?;
    data.truncate(n);
    if data.len() < 0x5000 {
        return None;
    }

    let mut i = 0x40usize;
    let mut stitch_chunks: Vec<(u64, u64)> = Vec::new();
    let mut stitch_buf: Vec<u8> = Vec::new();
    let mut files: Vec<(u64, Vec<(u64, u64)>, Vec<u8>)> = Vec::new();
    let mut decomp = Decompress::new(true);
    let mut out = vec![0u8; 131072];

    while i + 8 < data.len() {
        if is_zlib_hdr(&data[i..]) {
            decomp.reset(true);
            let end = (i + 512 * 1024).min(data.len());
            if let Ok(Status::StreamEnd) = decomp.decompress(&data[i..end], &mut out, FlushDecompress::None) {
                let out_len = decomp.total_out() as usize;
                let clen = decomp.total_in() as usize;
                if out_len >= 16 && clen > 0 {
                    stitch_chunks.push((i as u64, clen as u64));
                    stitch_buf.extend_from_slice(&out[..out_len]);
                    if out_len != 65536 {
                        let start_off = stitch_chunks[0].0;
                        files.push((start_off, std::mem::take(&mut stitch_chunks), std::mem::take(&mut stitch_buf)));
                    }
                    i = (i + clen + 15) & !15;
                    continue;
                }
            }
        }
        i += 16;
    }

    if files.is_empty() {
        return None;
    }

    // 1. Collect all ASCII identifier strings across the metadata files to resolve JOAAT hashes
    let mut hash_dict: HashMap<u32, String> = HashMap::new();
    for (_, _, payload) in &files {
        let mut cur = Vec::new();
        for &b in payload {
            if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'/' | b':' | b'-') {
                cur.push(b);
            } else {
                if cur.len() >= 3 {
                    if let Ok(s) = std::str::from_utf8(&cur) {
                        hash_dict.entry(joaat(s)).or_insert_with(|| s.to_string());
                        if let Some(stripped) = s.strip_prefix("Core.").or_else(|| s.strip_prefix("core.")) {
                            hash_dict.entry(joaat(stripped)).or_insert_with(|| stripped.to_string());
                        }
                        // every sub range of the dotted parts is hashed long runs would be quadratic
                        if s.contains('.') && s.len() <= 128 && s.bytes().filter(|&c| c == b'.').count() < 8 {
                            let parts: Vec<&str> = s.split('.').collect();
                            for a in 0..parts.len() {
                                for b_idx in (a + 1)..=parts.len() {
                                    let sub = parts[a..b_idx].join(".");
                                    hash_dict.entry(joaat(&sub)).or_insert(sub);
                                }
                            }
                        }
                    }
                }
                cur.clear();
            }
        }
    }

    // 2. Build metadata_chunks list for the UI file browser and locate BankSizes in DAT26
    let mut metadata_chunks = Vec::new();
    let mut raw_banks: Vec<(u32, u64, u64, u32)> = Vec::new();
    let mut bank_labels: HashMap<u32, Vec<String>> = HashMap::new();

    for (z_idx, (off, chunks, payload)) in files.iter().enumerate() {
        let num = z_idx + 1;
        let (kind, name) = if let Some(ver) = probe_dat(payload) {
            let tag = match (num, ver) {
                (1, 26) => "waveslots2",
                (3, 26) => "audioconfig",
                (5, 41) => "categories",
                (7, 25) => "curves",
                (8, 61) => "dynamix",
                (9, 581) => "sounds_common",
                (10, 86) => "music_common",
                (11, 20) => "effects",
                (14, 166) => "game_common",
                (15, 18) => "speech2_common",
                (18, 18) => "speechscanner",
                _ => "aud_meta",
            };
            ("DAT", format!("{num:04}_{tag}_dat{ver}.rel"))
        } else if payload.starts_with(b"AMB_STREAM_") {
            ("XML", format!("{num:04}_audio_packlist.txt"))
        } else if payload.starts_with(b"driverSettings_") {
            ("XML", format!("{num:04}_audio_driver_settings.txt"))
        } else if payload.starts_with(b"BaseCommon:") {
            ("XML", format!("{num:04}_audio_mounts.txt"))
        } else if payload.starts_with(b"core.ambience.") {
            ("XML", format!("{num:04}_audio_radiostations_and_packs.txt"))
        } else {
            ("ZLIB", format!("{num:04}_audio_meta.bin"))
        };
        metadata_chunks.push((name, kind, *off, payload.len() as u64, chunks.clone()));

        // Check if this DAT file has an Object Index Table (e.g. File #03 audioconfig DAT26)
        if probe_dat(payload).is_some() && payload.len() >= 16 {
            let dsz = rd32(payload, 4) as usize;
            let idx_off = 16usize.saturating_add(dsz);
            if idx_off + 12 <= payload.len() {
                let ntab = rd32(payload, idx_off);
                let ecnt = rd32(payload, idx_off + 8) as usize;
                if ntab == 4 && (1..=100_000).contains(&ecnt) && idx_off + 12 + ecnt * 8 <= payload.len() {
                    let h_banksizes = joaat("BankSizes");
                    let mut objs: Vec<(usize, u32)> = Vec::with_capacity(ecnt);
                    for k in 0..ecnt {
                        let eo = idx_off + 12 + k * 8;
                        let h = rd32(payload, eo);
                        let oo = 16usize.saturating_add(rd32(payload, eo + 4) as usize);
                        if oo < idx_off {
                            objs.push((oo, h));
                        }
                    }
                    objs.sort_by_key(|x| x.0);
                    for (oi, &(oo, h)) in objs.iter().enumerate() {
                        let nxt = objs.get(oi + 1).map(|x| x.0).unwrap_or(idx_off);
                        if h == h_banksizes && oo + 4 <= nxt {
                            let cnt = rd32(payload, oo) as usize;
                            if cnt > raw_banks.len() && cnt <= 20_000 && oo + 4 + cnt * 24 <= nxt {
                                raw_banks.clear();
                                for bk in 0..cnt {
                                    let bo = oo + 4 + bk * 24;
                                    let bh = rd32(payload, bo);
                                    let sz1 = rd32(payload, bo + 4) as u64;
                                    let pack_h = rd32(payload, bo + 16);
                                    let aligned = (sz1 + 4095) & !4095;
                                    raw_banks.push((bh, sz1, aligned, pack_h));
                                }
                            }
                        }
                    }
                    // Second pass over objects in DAT26 to extract WaveSlot -> bank_hash labels
                    if !raw_banks.is_empty() {
                        let bank_set: HashMap<u32, usize> = raw_banks.iter().enumerate().map(|(idx, &(bh, _, _, _))| (bh, idx)).collect();
                        for (oi, &(oo, h)) in objs.iter().enumerate() {
                            if h == h_banksizes {
                                continue;
                            }
                            let nxt = objs.get(oi + 1).map(|x| x.0).unwrap_or(idx_off);
                            if let Some(obj_name) = hash_dict.get(&h) {
                                let clean = obj_name
                                    .strip_prefix("Core.Resident.")
                                    .or_else(|| obj_name.strip_prefix("Core."))
                                    .or_else(|| obj_name.strip_prefix("core."))
                                    .unwrap_or(obj_name);
                                let mut p = oo;
                                while p + 4 <= nxt {
                                    let v = rd32(payload, p);
                                    if bank_set.contains_key(&v) {
                                        let list = bank_labels.entry(v).or_default();
                                        if list.len() < 16 && !list.iter().any(|s| s == clean) {
                                            list.push(clean.to_string());
                                        }
                                    }
                                    p += 4;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if raw_banks.is_empty() {
        return Some(AudioRpfCatalog {
            banks: Vec::new(),
            metadata_chunks,
        });
    }

    // 3. Scan length-prefixed sound strings in DAT166 / DAT581 and link nearby bank_hash references
    let bank_set: HashMap<u32, usize> = raw_banks.iter().enumerate().map(|(idx, &(bh, _, _, _))| (bh, idx)).collect();
    for (_, _, payload) in &files {
        if payload.len() < 1024 {
            continue;
        }
        // Find length-prefixed null-terminated strings: [len_with_null: u8] [ascii chars...] [0x00]
        let mut pos = 1usize;
        while pos + 5 < payload.len() {
            let slen = payload[pos - 1] as usize;
            if (4..=64).contains(&slen) && pos + slen <= payload.len() && payload[pos + slen - 1] == 0 {
                let sbytes = &payload[pos..pos + slen - 1];
                if sbytes[0].is_ascii_alphabetic()
                    && sbytes.iter().all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
                    && sbytes.contains(&b'_')
                {
                    if let Ok(sname) = std::str::from_utf8(sbytes) {
                        let win_end = (pos + slen + 160).min(payload.len());
                        let mut wp = pos + slen;
                        while wp + 4 <= win_end {
                            let v = rd32(payload, wp);
                            if bank_set.contains_key(&v) {
                                let list = bank_labels.entry(v).or_default();
                                if list.len() < 16 && !list.iter().any(|s| s == sname) {
                                    list.push(sname.to_string());
                                }
                            }
                            wp += 1;
                        }
                    }
                    pos += slen;
                    continue;
                }
            }
            pos += 1;
        }
    }

    let banks = raw_banks
        .into_iter()
        .map(|(bh, exact_size, aligned_size, ph)| {
            let pack_name = known_pack_fallback(ph)
                .map(|s| s.to_string())
                .or_else(|| hash_dict.get(&ph).cloned())
                .unwrap_or_else(|| format!("pack_{ph:08x}"));
            let used_by = bank_labels.remove(&bh).unwrap_or_default();
            AudioBankMeta {
                bank_hash: bh,
                exact_size,
                aligned_size,
                pack_name,
                used_by,
            }
        })
        .collect();

    Some(AudioRpfCatalog {
        banks,
        metadata_chunks,
    })
}

/// Aligns `BankSizes` metadata entries with the discovered `ADAT` (`.awc`) entries in `audio.rpf`
/// and returns descriptive names for each AWC index.
pub fn resolve_awc_names_from_catalog(
    awc_offsets_and_tracks: &[(u64, u32)],
    rpf_total_size: u64,
    catalog: &AudioRpfCatalog,
) -> Vec<Option<ResolvedAwcMeta>> {
    let n_awc = awc_offsets_and_tracks.len();
    let mut out = vec![None; n_awc];
    if catalog.banks.is_empty() || n_awc == 0 {
        return out;
    }

    let mut real_gaps = Vec::with_capacity(n_awc);
    for k in 0..n_awc {
        let nxt = if k + 1 < n_awc {
            awc_offsets_and_tracks[k + 1].0
        } else {
            rpf_total_size
        };
        real_gaps.push(nxt.saturating_sub(awc_offsets_and_tracks[k].0));
    }

    let mut by_sz: HashMap<u64, Vec<usize>> = HashMap::new();
    for (idx, &g) in real_gaps.iter().enumerate() {
        by_sz.entry(g).or_default().push(idx);
    }

    let bs = &catalog.banks;
    let mut cur_awc = 0usize;
    let mut matched_awc: Vec<Option<usize>> = vec![None; n_awc];

    for (bi, b) in bs.iter().enumerate() {
        let asz = b.aligned_size;
        let mut picked = None;

        for c in cur_awc..n_awc.min(cur_awc + 15) {
            if real_gaps[c] == asz {
                picked = Some(c);
                break;
            }
        }
        if picked.is_none() {
            let mut skipped_bs = false;
            for next_bi in (bi + 1)..bs.len().min(bi + 5) {
                for c in cur_awc..n_awc.min(cur_awc + 10) {
                    if real_gaps[c] == bs[next_bi].aligned_size {
                        skipped_bs = true;
                        break;
                    }
                }
                if skipped_bs {
                    break;
                }
            }
            if skipped_bs {
                continue;
            }
            for c in cur_awc..n_awc.min(cur_awc + 5) {
                if real_gaps[c].abs_diff(asz) <= 16384 {
                    picked = Some(c);
                    break;
                }
                if real_gaps[c] > asz
                    && bi + 1 < bs.len()
                    && c + 1 < n_awc
                    && real_gaps[c + 1] == bs[bi + 1].aligned_size
                {
                    picked = Some(c);
                    break;
                }
            }
        }
        if picked.is_none() && bi + 2 < bs.len() {
            let asz1 = bs[bi + 1].aligned_size;
            let asz2 = bs[bi + 2].aligned_size;
            if let Some(cands) = by_sz.get(&asz) {
                for &c in cands {
                    if c >= cur_awc && c + 2 < n_awc && real_gaps[c + 1] == asz1 && real_gaps[c + 2] == asz2 {
                        picked = Some(c);
                        break;
                    }
                }
            }
        }
        if let Some(c) = picked {
            matched_awc[c] = Some(bi);
            cur_awc = c + 1;
        }
    }

    // Format descriptive names and metadata for matched AWCs
    for k in 0..n_awc {
        let sc = awc_offsets_and_tracks[k].1;
        let idx_num = k + 1;
        if let Some(bi) = matched_awc[k] {
            let b = &bs[bi];
            let pack_slug = b.pack_name.replace('.', "_");
            let name = if let Some(sl) = b.used_by.first() {
                let sl_clean = sl.replace('.', "_");
                format!("sound_{idx_num:04}_{pack_slug}__{sl_clean}_{sc}st.awc")
            } else {
                format!("sound_{idx_num:04}_{pack_slug}_{sc}st.awc")
            };
            out[k] = Some(ResolvedAwcMeta {
                name,
                exact_size: b.exact_size,
                bank_hash: b.bank_hash,
                pack_name: b.pack_name.clone(),
                used_by: b.used_by.clone(),
            });
        }
    }
    out
}

// decode rage 9 audMetadata rel header hash index table, resolved names, and embedded identifiers
pub fn parse_dat(data: &[u8]) -> String {
    if data.len() < 16 {
        return String::from("Truncated DAT audio metadata");
    }
    let ver = rd32(data, 0);
    let data_sz = rd32(data, 4) as usize;
    let ts = rd32(data, 8);
    let flags = rd32(data, 12);

    // Collect clean identifier strings across the full DAT file for JOAAT resolution + summary
    let mut hash_dict: HashMap<u32, String> = HashMap::new();
    let mut clean_strs: Vec<String> = Vec::new();
    let mut seen_strs = std::collections::HashSet::new();
    let mut cur = Vec::new();
    for &b in data {
        if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'/' | b':' | b'-') {
            cur.push(b);
        } else {
            if cur.len() >= 4 {
                if let Ok(s) = std::str::from_utf8(&cur) {
                    if s.as_bytes()[0].is_ascii_alphabetic() {
                        hash_dict.entry(joaat(s)).or_insert_with(|| s.to_string());
                        if (s.contains('_') || s.contains('.') || s.contains('/'))
                            && clean_strs.len() < 500
                            && seen_strs.insert(s.to_string())
                        {
                            clean_strs.push(s.to_string());
                        }
                    }
                }
            }
            cur.clear();
        }
    }

    let mut out = String::with_capacity(8192);
    let _ = writeln!(
        out,
        "=== RAGE 9 Audio Metadata (DAT{ver}) ===\nTotal: {} B | Payload: {} B (0x{:x}) | BuildTag: 0x{:08x} | Flags: 0x{:08x}\n",
        data.len(),
        data_sz,
        data_sz,
        ts,
        flags
    );

    let idx_off = 16usize.saturating_add(data_sz);
    if idx_off + 12 <= data.len() {
        let num_tables = rd32(data, idx_off);
        let entry_cnt = rd32(data, idx_off + 8) as usize;
        let _ = writeln!(
            out,
            "--- Object Index Table @0x{idx_off:x} (tables={num_tables}, entries={entry_cnt}) ---"
        );
        let list_off = idx_off + 12;
        for i in 0..entry_cnt.min(250) {
            let e = list_off + i * 8;
            if e + 8 > data.len() {
                break;
            }
            let hash = rd32(data, e);
            let obj_off = rd32(data, e + 4);
            if let Some(name) = hash_dict.get(&hash) {
                let _ = writeln!(out, "  [{i:03}] {name} (0x{hash:08x}) -> obj@0x{obj_off:06x}");
            } else {
                let _ = writeln!(out, "  [{i:03}] hash=0x{hash:08x} -> obj@0x{obj_off:06x}");
            }
        }
        if entry_cnt > 250 {
            let _ = writeln!(out, "  ... ({} more entries)", entry_cnt - 250);
        }
    }

    if !clean_strs.is_empty() {
        let _ = writeln!(out, "\n--- Embedded Audio Identifiers / Paths ({}) ---", clean_strs.len());
        for s in clean_strs {
            let _ = writeln!(out, "  {s}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_and_parses_synthetic_dat26() {
        let mut buf = vec![0u8; 64];
        buf[0..4].copy_from_slice(&26u32.to_le_bytes());
        buf[4..8].copy_from_slice(&128u32.to_le_bytes());
        buf[8..12].copy_from_slice(&0x045a_88eau32.to_le_bytes());
        buf[12..16].copy_from_slice(&8u32.to_le_bytes());
        assert_eq!(probe_dat(&buf), Some(26));
        let txt = parse_dat(&buf);
        assert!(txt.contains("DAT26"));
        assert!(txt.contains("0x045a88ea"));
    }

    #[test]
    fn parses_real_audio_rpf_catalog_if_present() {
        let p = std::path::Path::new(r"C:\Games\GTA6.Xbox.v1.0.2.1-CYBERLEEK\audio.rpf");
        if !p.exists() {
            return;
        }
        let cat = parse_audio_rpf_catalog(p, 19_509_248).expect("should parse catalog");
        eprintln!("metadata_chunks={} banks={}", cat.metadata_chunks.len(), cat.banks.len());
        assert!(!cat.metadata_chunks.is_empty());
        assert!(!cat.banks.is_empty());
    }
}
