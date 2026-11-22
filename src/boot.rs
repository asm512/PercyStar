use std::fmt::Write as _;

fn rd16(d: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([d[o], d[o + 1]])
}

fn rd32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

// inspect pe64 executable for bringup: sections, console/gdk imports, and key anchors
pub fn inspect_pe64_for_bringup(data: &[u8]) -> String {
    if data.len() < 0x100 || &data[0..2] != b"MZ" {
        return String::from("Not a valid MZ/PE binary");
    }
    let lfanew = rd32(data, 0x3c) as usize;
    if lfanew + 0x108 > data.len() || &data[lfanew..lfanew + 4] != b"PE\0\0" {
        return String::from("Invalid PE header");
    }
    let machine = rd16(data, lfanew + 4);
    let nsec = rd16(data, lfanew + 6) as usize;
    let optsz = rd16(data, lfanew + 20) as usize;
    let magic = rd16(data, lfanew + 24);
    let dd_off = lfanew + 24 + if magic == 0x20b { 112 } else { 96 };
    let imp_rva = if dd_off + 16 <= data.len() { rd32(data, dd_off + 8) } else { 0 };
    let sec_off = lfanew + 24 + optsz;

    let mut secs = Vec::with_capacity(nsec);
    for i in 0..nsec {
        let o = sec_off + i * 40;
        if o + 40 > data.len() {
            break;
        }
        let name = String::from_utf8_lossy(&data[o..o + 8]).trim_end_matches('\0').to_string();
        let vsz = rd32(data, o + 8);
        let va = rd32(data, o + 12);
        let rsz = rd32(data, o + 16);
        let ra = rd32(data, o + 20);
        secs.push((name, va, vsz.max(rsz), ra, rsz));
    }

    let rva2off = |rva: u32| -> Option<usize> {
        for (_, va, vsz, ra, _) in &secs {
            if rva >= *va && rva < va.saturating_add(*vsz) {
                return ra.checked_add(rva - *va).map(|o| o as usize);
            }
        }
        None
    };

    let mut out = String::with_capacity(2048);
    let _ = writeln!(
        out,
        "=== PE64 Bring-Up Inspector ===\nSize: {} B | Machine: 0x{machine:04x} | Sections: {}",
        data.len(),
        secs.len()
    );
    for (name, va, _, ra, rsz) in &secs {
        let _ = writeln!(out, "  sec {name:<8} va=0x{va:08x} ra=0x{ra:08x} raw={rsz} B");
    }

    // parse import directory dlls
    if imp_rva != 0 {
        if let Some(mut ioff) = rva2off(imp_rva) {
            let mut dlls = Vec::new();
            // truncated or crafted images point names past the file and can list imports without end
            while ioff + 20 <= data.len() && dlls.len() < 4096 {
                let name_rva = rd32(data, ioff + 12);
                if name_rva == 0 {
                    break;
                }
                if let Some(tail) = rva2off(name_rva).and_then(|noff| data.get(noff..)) {
                    let tail = &tail[..tail.len().min(256)];
                    let end = tail.iter().position(|&b| b == 0).unwrap_or(0);
                    dlls.push(String::from_utf8_lossy(&tail[..end]).into_owned());
                }
                ioff += 20;
            }
            let _ = writeln!(out, "\n--- Imported DLLs ({}) ---", dlls.len());
            for dll in &dlls {
                let tag = if dll.to_ascii_lowercase().contains("xbox")
                    || dll.to_ascii_lowercase().contains("console")
                    || dll.to_ascii_lowercase().contains("gdk")
                    || dll.to_ascii_lowercase().contains("xgame")
                    || dll.to_ascii_lowercase().contains("d3d12")
                    || dll.to_ascii_lowercase().contains("kernelx")
                {
                    " [SHIM NEEDED]"
                } else {
                    ""
                };
                let _ = writeln!(out, "  {dll}{tag}");
            }
        }
    }

    // scan for key/format string anchors
    let _ = writeln!(out, "\n--- Format & Key String Anchors ---");
    for pat in [&b"9FPR"[..], b"RPF9", b"fiPackfile", b"scrProgram", b"0DRPz", b"Catalog000"] {
        let mut hits = 0usize;
        let mut first = None;
        let mut pos = 0usize;
        while let Some(rel) = data[pos..].windows(pat.len()).position(|w| w == pat) {
            let abs = pos + rel;
            if first.is_none() {
                first = Some(abs);
            }
            hits += 1;
            pos = abs + pat.len();
            if hits >= 16 {
                break;
            }
        }
        if let Some(f) = first {
            let _ = writeln!(out, "  {} -> {hits}+ hits (first @0x{f:x})", String::from_utf8_lossy(pat));
        }
    }
    out
}

// extract xml tag inner text helper
fn xml_val<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let s = xml.find(&open)? + open.len();
    let e = xml[s..].find(&close)?;
    Some(xml[s..s + e].trim())
}

// summarize xbox gdk MicrosoftGame.Config and layout xml
pub fn summarize_gdk_manifest(game_cfg: &str, layout_xml: Option<&str>) -> String {
    let mut out = String::with_capacity(1024);
    let _ = writeln!(out, "=== Xbox GDK Boot & Package Manifest ===");
    for tag in ["TitleId", "MSAAppId", "StoreId", "SMTEnabled", "SizeMB"] {
        if let Some(v) = xml_val(game_cfg, tag) {
            let _ = writeln!(out, "  {tag}: {v}");
        }
    }
    if let Some(pos) = game_cfg.find("Executable Name=\"") {
        let rest = &game_cfg[pos + 17..];
        if let Some(end) = rest.find('"') {
            let _ = writeln!(out, "  Executable: {}", &rest[..end]);
        }
    }
    if let Some(lay) = layout_xml {
        let chunks = lay.matches("<Chunk ").count();
        let rpfs = lay.matches(".rpf").count();
        let _ = writeln!(out, "  Layout Chunks: {chunks} | RPF Mount Refs: {rpfs}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_images_do_not_panic() {
        // every prefix of a minimal header shaped buffer and a few garbage fills
        let mut pe = vec![0u8; 0x400];
        pe[..2].copy_from_slice(b"MZ");
        pe[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        pe[0x80..0x84].copy_from_slice(b"PE\0\0");
        for fill in [0x00u8, 0x01, 0x7f, 0xff] {
            let mut img = pe.clone();
            for b in &mut img[0x84..] {
                *b = fill;
            }
            for len in [0x100usize, 0x101, 0x180, 0x200, 0x3ff, 0x400] {
                let _ = inspect_pe64_for_bringup(&img[..len]);
            }
        }
    }

    #[test]
    fn inspects_gdk_manifest_summary() {
        let cfg = r#"<Game><TitleId>1234ABCD</TitleId><Executable Name="game_console_master.exe" /><SizeMB>2048</SizeMB></Game>"#;
        let lay = r#"<Package><Chunk Id="1000"><FileGroup Include="common.rpf" /></Chunk></Package>"#;
        let s = summarize_gdk_manifest(cfg, Some(lay));
        assert!(s.contains("TitleId: 1234ABCD"));
        assert!(s.contains("Executable: game_console_master.exe"));
        assert!(s.contains("Layout Chunks: 1 | RPF Mount Refs: 1"));
    }

    #[test]
    fn inspects_minimal_pe64_and_anchors() {
        let mut pe = vec![0u8; 0x300];
        pe[0..2].copy_from_slice(b"MZ");
        pe[0x3c..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        pe[0x80..0x84].copy_from_slice(b"PE\0\0");
        pe[0x84..0x86].copy_from_slice(&0x8664u16.to_le_bytes());
        pe[0x86..0x88].copy_from_slice(&1u16.to_le_bytes());
        pe[0x94..0x96].copy_from_slice(&0xF0u16.to_le_bytes());
        pe[0x98..0x9a].copy_from_slice(&0x20bu16.to_le_bytes());
        let sec = 0x80 + 24 + 0xF0;
        pe[sec..sec + 5].copy_from_slice(b".text");
        pe[0x220..0x224].copy_from_slice(b"9FPR");
        let report = inspect_pe64_for_bringup(&pe);
        assert!(report.contains("Machine: 0x8664"));
        assert!(report.contains("9FPR -> 1+ hits"));
    }
}
