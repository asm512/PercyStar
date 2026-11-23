// typed xml view of a psin metadata file and the way back
// every editable value carries its file offset so applying an edited xml patches bytes in place
// and never moves or resizes anything
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::names;
use crate::psin::{joaat, parse_iff_chunks, parse_lenm, parse_lsch, PsinEnum, PsinField, PsinStruct};

const ELEMENT_NAME: u32 = 0x100;
// inline structs nest a dozen deep in real files, the cap only stops a schema that contains itself
const MAX_DEPTH: usize = 32;
// depth the fields of a block item are written at
const ITEM_DEPTH: usize = 3;

struct Layout<'a> {
    data: &'a [u8],
    // file offset and length of the instance payload
    payload: usize,
    payload_len: usize,
    // tag size and payload offset of every block
    blocks: Vec<(u32, usize, usize)>,
    structs: HashMap<u32, PsinStruct>,
    enums: HashMap<u32, PsinEnum>,
    schema: u32,
}

fn rd16(d: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([d[o], d[o + 1]])
}

fn rd32(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

fn rd64(d: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3], d[o + 4], d[o + 5], d[o + 6], d[o + 7]])
}

fn fnv1a(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c_9dc5u32, |h, &b| (h ^ b as u32).wrapping_mul(0x0100_0193))
}

fn layout(data: &[u8]) -> Result<Layout<'_>, String> {
    if data.len() < 16 || &data[..4] != b"PSIN" {
        return Err("Not a PSIN file".into());
    }
    let chunks = parse_iff_chunks(data);
    let body = chunks.first().filter(|c| c.0 == b"PSIN").map(|c| c.1).ok_or("PSIN section missing")?;
    let lsch = chunks.iter().find(|c| c.0 == b"LSCH").map(|c| c.1).ok_or("This PSIN variant has no LSCH schema")?;
    let lenm = chunks.iter().find(|c| c.0 == b"LENM").map(|c| c.1).unwrap_or(&[]);
    let mut i = 0usize;
    let mut table = Vec::new();
    loop {
        if i + 8 > body.len() {
            return Err("Instance table is not terminated".into());
        }
        if &body[i..i + 8] == b"pppppppp" {
            i += 8;
            break;
        }
        if i + 16 > body.len() {
            return Err("Instance table is cut short".into());
        }
        table.push((rd32(body, i), rd32(body, i + 4) as usize));
        i += 16;
    }
    let payload_len = body.len() - i;
    let mut blocks = Vec::with_capacity(table.len());
    let mut cursor = 0usize;
    for (tag, size) in table {
        if cursor.saturating_add(size) > payload_len {
            return Err("An instance runs past the end of the payload".into());
        }
        blocks.push((tag, size, cursor));
        cursor = cursor.saturating_add((size + 15) & !15);
    }
    Ok(Layout {
        data,
        payload: 8 + i,
        payload_len,
        blocks,
        structs: parse_lsch(lsch).into_iter().map(|s| (s.hash, s)).collect(),
        enums: parse_lenm(lenm).into_iter().map(|e| (e.hash, e)).collect(),
        schema: fnv1a(lsch),
    })
}

// every hashed struct, field and enum value name of the file with the value types of each field
pub fn collect_name_targets(data: &[u8], out: &mut names::NameTargets) {
    let Ok(l) = layout(data) else { return };
    for s in l.structs.values() {
        // small ids belong to unnamed structs
        if s.hash > 0xffff {
            out.structs.insert(s.hash);
        }
        for f in s.fields.iter().filter(|f| f.name_hash != ELEMENT_NAME) {
            *out.fields.entry(f.name_hash).or_default() |= 1u128 << (f.type_raw & 0x7f);
            out.members.entry(s.hash).or_default().insert(f.name_hash);
        }
    }
    for e in l.enums.values() {
        out.enums.entry(e.hash).or_default().extend(e.values.iter().map(|v| v.1));
    }
}

// text held in the file's string blocks, the only readable names a psin carries
pub fn collect_strings(data: &[u8], out: &mut HashSet<String>) {
    let Ok(l) = layout(data) else { return };
    for &(tag, size, off) in &l.blocks {
        if tag == 0x10 {
            names::harvest_tokens(&l.data[l.payload + off..l.payload + off + size], out);
        }
    }
}

// value types that are stored inline with a fixed size, xml type name and byte size
fn scalar(kind: u32, gap: usize) -> Option<(&'static str, usize)> {
    Some(match kind {
        0x01 => ("bool", 1),
        0x10 => ("s8", 1),
        0x11 => ("u8", 1),
        0x12 => ("s16", 2),
        0x13 => ("u16", 2),
        0x14 => ("s32", 4),
        0x15 => ("u32", 4),
        0x16 => ("s64", 8),
        0x17 => ("u64", 8),
        0x20 => ("f16", 2),
        0x21 => ("f32", 4),
        0x30 => ("vec2", 8),
        0x32 => ("vec3", 12),
        0x31 | 0x33 | 0x34 => ("vec4", 16),
        0x35 => ("mat3x4", 48),
        0x36 => ("mat4", 64),
        0x4a | 0x4b | 0x4d => ("hash", 4),
        0x08 | 0x4f => ("hash64", 8),
        0x60 => ("enum8", 1),
        0x61 => ("enum16", 2),
        0x62 => ("enum32", 4),
        0x64 => ("flags16", 2),
        // stored width follows the room the schema leaves
        0x63 | 0x65 => match gap {
            0 | 1 => ("flags8", 1),
            2 | 3 => ("flags16", 2),
            _ => ("flags32", 4),
        },
        _ => return None,
    })
}

fn float_count(ty: &str) -> Option<usize> {
    Some(match ty {
        "vec2" => 2,
        "vec3" => 3,
        "vec4" => 4,
        "mat3x4" => 12,
        "mat4" => 16,
        _ => return None,
    })
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // subnormal half becomes a normal float
            let shift = man.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((man << shift) & 0x3ff) << 13
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

fn f32_to_half(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 112;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - e);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    // rounding may carry into the exponent which is the right result
    (sign as u32 | ((e as u32) << 10) + ((man + 0x1000) >> 13)) as u16
}

// shortest text that parses back to the same bits, raw bits for anything that does not
fn fmt_f32(v: f32) -> String {
    let s = format!("{v}");
    if v.is_finite() && s.parse::<f32>().is_ok_and(|p| p.to_bits() == v.to_bits()) {
        s
    } else {
        format!("bits:0x{:08x}", v.to_bits())
    }
}

fn fmt_f16(h: u16) -> String {
    let v = half_to_f32(h);
    if v.is_finite() && f32_to_half(v) == h {
        format!("{v}")
    } else {
        format!("bits:0x{h:04x}")
    }
}

fn esc(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "&#{};", c as u32);
            }
            c => out.push(c),
        }
    }
}

// resolved name of a schema hash, a trailing question mark flags a name assembled from words
fn name_of(hash: u32) -> String {
    match names::lookup(hash) {
        Some(n) if names::is_built(hash) => format!("{n}?"),
        Some(n) => n,
        None => format!("0x{hash:08x}"),
    }
}

struct Writer<'a> {
    l: &'a Layout<'a>,
    out: String,
    // stop adding detail once the text is this long, zero means no limit
    limit: usize,
    cut: bool,
}

impl Writer<'_> {
    fn full(&mut self) -> bool {
        if self.limit != 0 && self.out.len() > self.limit {
            self.cut = true;
        }
        self.cut
    }

    fn pad(&mut self, depth: usize) {
        for _ in 0..depth {
            self.out.push_str("  ");
        }
    }

    // block index and offset a stored pointer refers to
    fn target(&self, ptr: u64) -> Option<(usize, usize)> {
        let idx = (ptr & 0xf_ffff) as usize;
        let off = (ptr >> 20) as usize;
        let &(_, size, _) = self.l.blocks.get(idx.checked_sub(1)?)?;
        (off <= size).then_some((idx - 1, off))
    }

    fn to_attr(&mut self, ptr: u64) {
        match self.target(ptr) {
            _ if ptr == 0 => self.out.push_str(" to=\"null\""),
            Some((b, o)) => {
                let _ = write!(self.out, " to=\"{b}+0x{o:x}\"");
            }
            None => {
                let _ = write!(self.out, " to=\"?0x{ptr:x}\"");
            }
        }
    }

    fn c_string(&self, ptr: u64, max: usize) -> Option<String> {
        let (b, o) = self.target(ptr)?;
        let (_, size, off) = self.l.blocks[b];
        let bytes = &self.l.data[self.l.payload + off + o..self.l.payload + off + size];
        let end = bytes.iter().take(max).position(|&c| c == 0).unwrap_or(bytes.len().min(max));
        Some(String::from_utf8_lossy(&bytes[..end]).into_owned())
    }

    // one inline value at a file offset, label is the leading attributes already formatted
    fn scalar(&mut self, depth: usize, label: &str, ty: &str, size: usize, pos: usize, enum_hash: u32) {
        let d = self.l.data;
        self.pad(depth);
        let _ = write!(self.out, "<f{label} t=\"{ty}\" o=\"0x{pos:x}\" v=\"");
        let mut extra = String::new();
        match ty {
            "bool" => self.out.push_str(match d[pos] {
                0 => "false",
                1 => "true",
                other => {
                    let _ = write!(extra, " raw=\"{other}\"");
                    "true"
                }
            }),
            "s8" => {
                let _ = write!(self.out, "{}", d[pos] as i8);
            }
            "u8" => {
                let _ = write!(self.out, "{}", d[pos]);
            }
            "s16" => {
                let _ = write!(self.out, "{}", rd16(d, pos) as i16);
            }
            "u16" => {
                let _ = write!(self.out, "{}", rd16(d, pos));
            }
            "s32" => {
                let _ = write!(self.out, "{}", rd32(d, pos) as i32);
            }
            "u32" => {
                let _ = write!(self.out, "{}", rd32(d, pos));
            }
            "s64" => {
                let _ = write!(self.out, "{}", rd64(d, pos) as i64);
            }
            "u64" => {
                let _ = write!(self.out, "{}", rd64(d, pos));
            }
            "f16" => self.out.push_str(&fmt_f16(rd16(d, pos))),
            "f32" => self.out.push_str(&fmt_f32(f32::from_bits(rd32(d, pos)))),
            "hash" => {
                let h = rd32(d, pos);
                let _ = write!(self.out, "0x{h:08x}");
                if let Some(n) = names::lookup_value(h) {
                    extra.push_str(" s=\"");
                    esc(&n, &mut extra);
                    extra.push('"');
                }
            }
            "hash64" => {
                let _ = write!(self.out, "0x{:016x}", rd64(d, pos));
            }
            "enum8" | "enum16" | "enum32" => {
                let v = match size {
                    1 => d[pos] as i8 as i64,
                    2 => rd16(d, pos) as i16 as i64,
                    _ => rd32(d, pos) as i32 as i64,
                };
                let _ = write!(self.out, "{v}");
                if let Some(nh) = self.l.enums.get(&enum_hash).and_then(|e| e.values.iter().find(|x| x.0 == v)).map(|x| x.1) {
                    let _ = write!(extra, " e=\"{}\"", name_of(nh));
                }
            }
            "flags8" => {
                let _ = write!(self.out, "0x{:02x}", d[pos]);
            }
            "flags16" => {
                let _ = write!(self.out, "0x{:04x}", rd16(d, pos));
            }
            "flags32" => {
                let _ = write!(self.out, "0x{:08x}", rd32(d, pos));
            }
            _ => {
                for k in 0..size / 4 {
                    if k != 0 {
                        self.out.push(' ');
                    }
                    self.out.push_str(&fmt_f32(f32::from_bits(rd32(d, pos + k * 4))));
                }
            }
        }
        self.out.push('"');
        self.out.push_str(&extra);
        self.out.push_str("/>\n");
    }

    // the value of one schema field, base is the file offset of the struct it belongs to
    #[allow(clippy::too_many_arguments)]
    fn field(&mut self, depth: usize, label: &str, f: &PsinField, fields: &[PsinField], gap: usize, pos: usize, end: usize) {
        if self.full() {
            return;
        }
        let d = self.l.data;
        let kind = f.type_raw & 0xff;
        if let Some((ty, size)) = scalar(kind, gap) {
            if pos + size <= end {
                self.scalar(depth, label, ty, size, pos, f.ref_hash);
            }
            return;
        }
        let elem = fields.get((f.type_raw >> 16) as usize).filter(|e| e.name_hash == ELEMENT_NAME);
        match kind {
            0x05 => {
                let Some(st) = self.l.structs.get(&f.ref_hash) else { return };
                self.pad(depth);
                let _ = writeln!(self.out, "<s{label} type=\"{}\">", name_of(f.ref_hash));
                if depth < MAX_DEPTH {
                    self.struct_fields(depth + 1, st, pos, end);
                }
                self.pad(depth);
                self.out.push_str("</s>\n");
            }
            0x07 if pos + 8 <= end => {
                self.pad(depth);
                let _ = write!(self.out, "<f{label} t=\"ptr\"");
                self.to_attr(rd64(d, pos));
                self.out.push_str("/>\n");
            }
            0x42 if pos + 8 <= end => {
                let ptr = rd64(d, pos);
                self.pad(depth);
                let _ = write!(self.out, "<f{label} t=\"string\"");
                self.to_attr(ptr);
                if let Some(s) = self.c_string(ptr, 4096) {
                    self.out.push_str(" text=\"");
                    esc(&s, &mut self.out);
                    self.out.push('"');
                }
                self.out.push_str("/>\n");
            }
            0x44 if pos + 12 <= end => {
                let ptr = rd64(d, pos);
                let len = rd16(d, pos + 8) as usize;
                self.pad(depth);
                let _ = write!(self.out, "<f{label} t=\"string\"");
                self.to_attr(ptr);
                if let Some(s) = self.c_string(ptr, len) {
                    self.out.push_str(" text=\"");
                    esc(&s, &mut self.out);
                    self.out.push('"');
                }
                self.out.push_str("/>\n");
            }
            0x52 if pos + 12 <= end => {
                self.pad(depth);
                let _ = write!(self.out, "<f{label} t=\"array\" count=\"{}\"", rd16(d, pos + 8));
                self.to_attr(rd64(d, pos));
                self.out.push_str("/>\n");
            }
            0x5c if pos + 16 <= end => {
                self.pad(depth);
                let _ = write!(self.out, "<f{label} t=\"array\" count=\"{}\"", rd32(d, pos));
                self.to_attr(rd64(d, pos + 8));
                self.out.push_str("/>\n");
            }
            0x50 if elem.is_some() => {
                let e = elem.unwrap();
                let count = f.ref_hash as usize;
                let esize = match scalar(e.type_raw & 0xff, usize::MAX) {
                    Some((_, s)) => s,
                    None if e.type_raw & 0xff == 0x05 => self.l.structs.get(&e.ref_hash).map_or(0, |s| s.size as usize),
                    None => 0,
                };
                if esize == 0 || count == 0 || count > 4096 || count * esize > gap {
                    self.raw(depth, label, kind, gap, pos, end);
                    return;
                }
                self.pad(depth);
                let _ = writeln!(self.out, "<a{label} count=\"{count}\">");
                for k in 0..count {
                    self.field(depth + 1, &format!(" i=\"{k}\""), e, fields, esize, pos + k * esize, end);
                }
                self.pad(depth);
                self.out.push_str("</a>\n");
            }
            _ => self.raw(depth, label, kind, gap, pos, end),
        }
    }

    fn raw(&mut self, depth: usize, label: &str, kind: u32, gap: usize, pos: usize, end: usize) {
        let n = gap.min(64).min(end.saturating_sub(pos));
        self.pad(depth);
        let _ = write!(self.out, "<f{label} t=\"raw:0x{kind:02x}\" hex=\"");
        for b in &self.l.data[pos..pos + n] {
            let _ = write!(self.out, "{b:02x}");
        }
        self.out.push_str("\"/>\n");
    }

    fn struct_fields(&mut self, depth: usize, st: &PsinStruct, base: usize, end: usize) {
        for f in st.fields.iter().filter(|f| f.name_hash != ELEMENT_NAME) {
            let off = f.offset as usize;
            // room up to the next field tells the stored width of types whose size varies
            let next = st.fields.iter().filter(|o| o.name_hash != ELEMENT_NAME).map(|o| o.offset as usize).filter(|&o| o > off).min();
            let gap = next.unwrap_or(st.size as usize).saturating_sub(off);
            let mut label = String::from(" n=\"");
            esc(&name_of(f.name_hash), &mut label);
            label.push('"');
            if base + off < end {
                self.field(depth, &label, f, &st.fields, gap, base + off, end);
            }
        }
    }

    fn block(&mut self, index: usize) {
        let (tag, size, off) = self.l.blocks[index];
        let start = self.l.payload + off;
        let end = start + size;
        let _ = write!(self.out, "  <block i=\"{index}\" size=\"{size}\"");
        if let Some(st) = self.l.structs.get(&tag) {
            let isize = (st.size as usize).max(1);
            let count = if st.size == 0 { 1 } else { size / isize };
            let _ = writeln!(self.out, " type=\"{}\" count=\"{count}\">", name_of(tag));
            for k in 0..count {
                if self.full() {
                    break;
                }
                let _ = writeln!(self.out, "    <item i=\"{k}\">");
                self.struct_fields(ITEM_DEPTH, st, start + k * isize, (start + (k + 1) * isize).min(end));
                self.out.push_str("    </item>\n");
            }
        } else if tag == 0x10 {
            self.out.push_str(" type=\"chars\">\n");
            for s in self.l.data[start..end].split(|&c| c == 0).filter(|s| !s.is_empty()) {
                if self.full() {
                    break;
                }
                self.out.push_str("    <text v=\"");
                esc(&String::from_utf8_lossy(s), &mut self.out);
                self.out.push_str("\"/>\n");
            }
        } else if tag == 0x07 {
            let _ = writeln!(self.out, " type=\"ptr\" count=\"{}\">", size / 8);
            for k in 0..size / 8 {
                if self.full() {
                    break;
                }
                let _ = write!(self.out, "    <f i=\"{k}\" t=\"ptr\"");
                self.to_attr(rd64(self.l.data, start + k * 8));
                self.out.push_str("/>\n");
            }
        } else if let Some((ty, esize)) = scalar(tag, usize::MAX).filter(|_| tag < 0x100) {
            let _ = writeln!(self.out, " type=\"{ty}\" count=\"{}\">", size / esize);
            for k in 0..size / esize {
                if self.full() {
                    break;
                }
                self.scalar(2, &format!(" i=\"{k}\""), ty, esize, start + k * esize, 0);
            }
        } else {
            let _ = writeln!(self.out, " type=\"raw:0x{tag:x}\">");
        }
        self.out.push_str("  </block>\n");
    }
}

// xml text of the whole file, limit caps the length for on screen previews and zero means everything
pub fn export_xml(data: &[u8], limit: usize) -> Result<String, String> {
    let l = layout(data)?;
    let mut w = Writer { l: &l, out: String::with_capacity(data.len().min(1 << 20) * 4), limit, cut: false };
    let _ = writeln!(
        w.out,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!-- edit the v values, keep t and o as they are: o is the byte offset the value is written back to -->
<!-- a hash may be given as text, it is hashed in lower case -->\n<psin size=\"{}\" schema=\"0x{:08x}\" blocks=\"{}\">",
        data.len(),
        l.schema,
        l.blocks.len()
    );
    for i in 0..l.blocks.len() {
        if w.full() {
            break;
        }
        w.block(i);
    }
    if w.cut {
        w.out.push_str("  <!-- preview cut here, export the xml to get the whole file -->\n");
    }
    w.out.push_str("</psin>\n");
    Ok(w.out)
}

fn unesc(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let Some(end) = rest.find(';') else { break };
        let ent = &rest[1..end];
        match ent {
            "amp" => out.push('&'),
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            _ => match ent.strip_prefix('#').and_then(|n| n.parse::<u32>().ok()).and_then(char::from_u32) {
                Some(c) => out.push(c),
                None => out.push_str(&rest[..=end]),
            },
        }
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

// attributes of the tag that starts at the given text, which ends at the first unquoted >
fn attrs(tag: &str) -> HashMap<&str, String> {
    let mut map = HashMap::new();
    let b = tag.as_bytes();
    let mut i = 0;
    while i < b.len() && b[i] != b'>' {
        if !(b[i].is_ascii_alphabetic()) {
            i += 1;
            continue;
        }
        let ks = i;
        while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
            i += 1;
        }
        let key = &tag[ks..i];
        if b.get(i) != Some(&b'=') || b.get(i + 1) != Some(&b'"') {
            continue;
        }
        let vs = i + 2;
        let Some(len) = tag[vs..].find('"') else { break };
        map.insert(key, unesc(&tag[vs..vs + len]));
        i = vs + len + 1;
    }
    map
}

fn parse_int(s: &str) -> Option<i128> {
    let s = s.trim();
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s),
    };
    let v = match body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        Some(h) => i128::from_str_radix(h, 16).ok()?,
        None => body.parse::<i128>().ok()?,
    };
    Some(if neg { -v } else { v })
}

fn parse_f32(s: &str) -> Option<f32> {
    let s = s.trim();
    match s.strip_prefix("bits:") {
        Some(b) => parse_int(b).and_then(|v| u32::try_from(v).ok()).map(f32::from_bits),
        None => s.parse::<f32>().ok(),
    }
}

// bytes a value of the given xml type is stored as
fn encode(ty: &str, v: &str) -> Result<Vec<u8>, String> {
    let bad = || format!("\"{v}\" is not a valid {ty}");
    let int = |min: i128, max: i128| parse_int(v).filter(|x| (min..=max).contains(x)).ok_or_else(bad);
    Ok(match ty {
        "bool" => vec![match v.trim() {
            "true" | "1" => 1,
            "false" | "0" => 0,
            _ => return Err(bad()),
        }],
        "s8" | "enum8" => vec![int(i8::MIN as i128, u8::MAX as i128)? as u8],
        "u8" | "flags8" => vec![int(0, u8::MAX as i128)? as u8],
        "s16" | "enum16" => (int(i16::MIN as i128, u16::MAX as i128)? as u16).to_le_bytes().to_vec(),
        "u16" | "flags16" => (int(0, u16::MAX as i128)? as u16).to_le_bytes().to_vec(),
        "s32" | "enum32" => (int(i32::MIN as i128, u32::MAX as i128)? as u32).to_le_bytes().to_vec(),
        "u32" | "flags32" => (int(0, u32::MAX as i128)? as u32).to_le_bytes().to_vec(),
        "s64" => (int(i64::MIN as i128, u64::MAX as i128)? as u64).to_le_bytes().to_vec(),
        "u64" | "hash64" => (int(0, u64::MAX as i128)? as u64).to_le_bytes().to_vec(),
        "f32" => parse_f32(v).ok_or_else(bad)?.to_bits().to_le_bytes().to_vec(),
        "f16" => match v.trim().strip_prefix("bits:") {
            Some(b) => (parse_int(b).filter(|x| (0..=0xffff).contains(x)).ok_or_else(bad)? as u16).to_le_bytes().to_vec(),
            None => f32_to_half(v.trim().parse::<f32>().map_err(|_| bad())?).to_le_bytes().to_vec(),
        },
        // a hash may be given as the name it stands for
        "hash" => match parse_int(v) {
            Some(x) => u32::try_from(x).map_err(|_| bad())?.to_le_bytes().to_vec(),
            None if names::is_name(v.trim()) => joaat(v.trim()).to_le_bytes().to_vec(),
            None => return Err(bad()),
        },
        _ => {
            let n = float_count(ty).ok_or_else(|| format!("type {ty} cannot be edited"))?;
            let parts: Vec<&str> = v.split_whitespace().collect();
            if parts.len() != n {
                return Err(format!("{ty} needs {n} numbers, got {}", parts.len()));
            }
            let mut out = Vec::with_capacity(n * 4);
            for p in parts {
                out.extend_from_slice(&parse_f32(p).ok_or_else(bad)?.to_bits().to_le_bytes());
            }
            out
        }
    })
}

pub struct Applied {
    pub data: Vec<u8>,
    // one line per value that differs from the file
    pub changes: Vec<String>,
}

// writes the values of an edited xml into a copy of the file it was exported from
pub fn apply_xml(data: &[u8], xml: &str) -> Result<Applied, String> {
    let l = layout(data)?;
    let root = xml.find("<psin ").ok_or("No <psin> element found")?;
    let head = attrs(&xml[root + 5..]);
    let size_ok = head.get("size").and_then(|s| s.parse::<usize>().ok()) == Some(data.len());
    let schema_ok = head.get("schema").and_then(|s| parse_int(s)) == Some(l.schema as i128);
    if !size_ok || !schema_ok {
        return Err("This XML was exported from a different file (size or schema differs)".into());
    }
    let (lo, hi) = (l.payload, l.payload + l.payload_len);
    let mut out = data.to_vec();
    let mut changes = Vec::new();
    let mut rest = &xml[root..];
    while let Some(i) = rest.find("<f ") {
        rest = &rest[i + 2..];
        let a = attrs(rest);
        let (Some(ty), Some(o), Some(v)) = (a.get("t"), a.get("o"), a.get("v")) else { continue };
        let label = a.get("n").or(a.get("i")).cloned().unwrap_or_default();
        let pos = parse_int(o).and_then(|p| usize::try_from(p).ok()).ok_or_else(|| format!("{label}: bad offset {o}"))?;
        write_value(&mut out, (lo, hi), &label, ty, pos, v, &mut changes)?;
    }
    Ok(Applied { data: out, changes })
}

// stores one value at its file offset and notes the change when the bytes differ
fn write_value(out: &mut [u8], (lo, hi): (usize, usize), label: &str, ty: &str, pos: usize, v: &str, changes: &mut Vec<String>) -> Result<(), String> {
    let bytes = encode(ty, v).map_err(|e| format!("{label} at 0x{pos:x}: {e}"))?;
    if pos < lo || pos + bytes.len() > hi {
        return Err(format!("{label}: offset 0x{pos:x} is outside the data"));
    }
    if out[pos..pos + bytes.len()] != bytes[..] {
        // a bool stored as something other than 0 or 1 reads as true and is left alone
        if ty == "bool" && bytes[0] == 1 && out[pos] != 0 {
            return Ok(());
        }
        changes.push(format!("{label} ({ty}) at 0x{pos:x} = {v}"));
        out[pos..pos + bytes.len()].copy_from_slice(&bytes);
    }
    Ok(())
}

// one line of the tree view
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Value {
    pub label: String,
    pub ty: String,
    // file offset of a value that can be edited in place
    pub offset: Option<usize>,
    pub text: String,
    // what the value stands for: an enum value name, the text behind a hash
    pub note: String,
    // block and byte offset a pointer leads to
    pub to: Option<(usize, usize)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Value(Value),
    Group { label: String, kind: String, children: Vec<Node> },
}

pub struct BlockInfo {
    pub label: String,
    // items of a struct block, values of a plain one, strings of a text block
    pub count: usize,
    pub item_size: usize,
    pub size: usize,
}

// the blocks of a file without their content
pub fn outline(data: &[u8]) -> Result<Vec<BlockInfo>, String> {
    let l = layout(data)?;
    Ok(l.blocks
        .iter()
        .map(|&(tag, size, off)| {
            let (label, count, item_size) = if let Some(st) = l.structs.get(&tag) {
                let isize = (st.size as usize).max(1);
                (name_of(tag), if st.size == 0 { 1 } else { size / isize }, isize)
            } else if tag == 0x10 {
                let bytes = &data[l.payload + off..l.payload + off + size];
                ("chars".to_string(), bytes.split(|&c| c == 0).filter(|s| !s.is_empty()).count(), 0)
            } else if tag == 0x07 {
                ("ptr".to_string(), size / 8, 8)
            } else if let Some((ty, esize)) = scalar(tag, usize::MAX).filter(|_| tag < 0x100) {
                (ty.to_string(), size / esize, esize)
            } else {
                (format!("raw:0x{tag:x}"), 0, 0)
            };
            BlockInfo { label, count, item_size, size }
        })
        .collect())
}

fn parse_target(to: &str) -> Option<(usize, usize)> {
    let (block, off) = to.split_once('+')?;
    Some((block.parse().ok()?, usize::try_from(parse_int(off)?).ok()?))
}

fn node_label(a: &HashMap<&str, String>) -> String {
    a.get("n").cloned().or_else(|| a.get("i").map(|i| format!("[{i}]"))).unwrap_or_default()
}

// the lines the xml writer produced for one item as tree nodes
fn parse_nodes(text: &str) -> Vec<Node> {
    let mut stack: Vec<(String, String, Vec<Node>)> = vec![(String::new(), String::new(), Vec::new())];
    let close = |stack: &mut Vec<(String, String, Vec<Node>)>| {
        let (label, kind, children) = stack.pop().unwrap();
        stack.last_mut().unwrap().2.push(Node::Group { label, kind, children });
    };
    for line in text.lines().map(str::trim_start) {
        if line.starts_with("</") {
            if stack.len() > 1 {
                close(&mut stack);
            }
        } else if let Some(rest) = line.strip_prefix("<s") {
            let a = attrs(rest);
            stack.push((node_label(&a), a.get("type").cloned().unwrap_or_default(), Vec::new()));
        } else if let Some(rest) = line.strip_prefix("<a") {
            let a = attrs(rest);
            stack.push((node_label(&a), format!("{} values", a.get("count").map_or("?", |c| c.as_str())), Vec::new()));
        } else if let Some(rest) = line.strip_prefix("<f") {
            let a = attrs(rest);
            let ty = a.get("t").cloned().unwrap_or_default();
            let mut v = Value {
                label: node_label(&a),
                offset: a.get("o").and_then(|o| parse_int(o)).and_then(|o| usize::try_from(o).ok()),
                text: a.get("v").or(a.get("text")).or(a.get("hex")).cloned().unwrap_or_default(),
                note: a.get("e").or(a.get("s")).cloned().unwrap_or_default(),
                ..Value::default()
            };
            if ty == "array" {
                v.text = format!("{} values", a.get("count").map_or("?", |c| c.as_str()));
            }
            match a.get("to").map(String::as_str) {
                Some("null") => v.note = "null".into(),
                Some(to) => match parse_target(to) {
                    Some(t) => v.to = Some(t),
                    None => v.note = "pointer leads nowhere".into(),
                },
                None => {}
            }
            v.ty = ty;
            stack.last_mut().unwrap().2.push(Node::Value(v));
        }
    }
    while stack.len() > 1 {
        close(&mut stack);
    }
    stack.pop().unwrap().2
}

// tree nodes of the given items of one block, the same values the xml shows
pub fn items(data: &[u8], block: usize, first: usize, count: usize) -> Result<Vec<Vec<Node>>, String> {
    let l = layout(data)?;
    let &(tag, size, off) = l.blocks.get(block).ok_or("No such block")?;
    let start = l.payload + off;
    let end = start + size;
    let mut w = Writer { l: &l, out: String::new(), limit: 0, cut: false };
    let mut out = Vec::new();
    let last = first.saturating_add(count);
    if let Some(st) = l.structs.get(&tag) {
        let isize = (st.size as usize).max(1);
        let total = if st.size == 0 { 1 } else { size / isize };
        for k in first..last.min(total) {
            w.out.clear();
            w.struct_fields(ITEM_DEPTH, st, start + k * isize, (start + (k + 1) * isize).min(end));
            out.push(parse_nodes(&w.out));
        }
    } else if tag == 0x10 {
        for (k, s) in data[start..end].split(|&c| c == 0).filter(|s| !s.is_empty()).enumerate().skip(first).take(count) {
            out.push(vec![Node::Value(Value { label: format!("[{k}]"), ty: "text".into(), text: String::from_utf8_lossy(s).into_owned(), ..Value::default() })]);
        }
    } else if tag == 0x07 {
        for k in first..last.min(size / 8) {
            w.out.clear();
            let _ = write!(w.out, "<f i=\"{k}\" t=\"ptr\"");
            w.to_attr(rd64(data, start + k * 8));
            w.out.push_str("/>\n");
            out.push(parse_nodes(&w.out));
        }
    } else if let Some((ty, esize)) = scalar(tag, usize::MAX).filter(|_| tag < 0x100) {
        for k in first..last.min(size / esize) {
            w.out.clear();
            w.scalar(0, &format!(" i=\"{k}\""), ty, esize, start + k * esize, 0);
            out.push(parse_nodes(&w.out));
        }
    }
    Ok(out)
}

// whether the text is a valid value of the type
pub fn check(ty: &str, v: &str) -> Result<(), String> {
    encode(ty, v).map(|_| ())
}

// writes edited values, given as file offset, type, text and label, into a copy of the file
pub fn patch<'a>(data: &[u8], edits: impl IntoIterator<Item = (usize, &'a str, &'a str, &'a str)>) -> Result<Applied, String> {
    let l = layout(data)?;
    let bounds = (l.payload, l.payload + l.payload_len);
    let mut out = data.to_vec();
    let mut changes = Vec::new();
    for (pos, ty, v, label) in edits {
        write_value(&mut out, bounds, label, ty, pos, v, &mut changes)?;
    }
    Ok(Applied { data: out, changes })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn section(tag: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut v = tag.to_vec();
        v.extend_from_slice(&((body.len() + 8) as u32).to_be_bytes());
        v.extend_from_slice(body);
        v
    }

    fn field(name: u32, offset: u32, ty: u32, reference: u32) -> Vec<u8> {
        [name, offset, ty, reference].iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    // one struct holding one of each common value, a second block of floats it points at
    pub(crate) fn sample() -> Vec<u8> {
        let st = joaat("TestThing");
        let mut item = vec![0u8; 64];
        item[8..12].copy_from_slice(&1630.0f32.to_bits().to_le_bytes());
        item[12..16].copy_from_slice(&(-7i32).to_le_bytes());
        item[16..20].copy_from_slice(&joaat("cheetah2").to_le_bytes());
        item[20] = 1;
        item[22..24].copy_from_slice(&0x3c00u16.to_le_bytes());
        for (k, v) in [1.0f32, 2.0, 3.0, 0.5].iter().enumerate() {
            item[32 + k * 4..36 + k * 4].copy_from_slice(&v.to_bits().to_le_bytes());
        }
        // array of two floats in block 1
        item[48..56].copy_from_slice(&2u64.to_le_bytes());
        item[56..58].copy_from_slice(&2u16.to_le_bytes());
        item[58..60].copy_from_slice(&2u16.to_le_bytes());
        let mut floats = Vec::new();
        floats.extend_from_slice(&0.25f32.to_bits().to_le_bytes());
        floats.extend_from_slice(&f32::NAN.to_bits().to_le_bytes());

        let mut body = Vec::new();
        for (tag, size) in [(st, 64u32), (0x21, 8)] {
            body.extend_from_slice(&tag.to_le_bytes());
            body.extend_from_slice(&size.to_le_bytes());
            body.extend_from_slice(&[0xAA; 8]);
        }
        body.extend_from_slice(b"pppppppp");
        body.extend_from_slice(&item);
        body.extend_from_slice(&floats);
        body.extend_from_slice(&[0u8; 8]);

        let mut lsch = Vec::new();
        lsch.extend_from_slice(&st.to_le_bytes());
        lsch.extend_from_slice(&[0u8; 12]);
        lsch.extend_from_slice(&[0xAA; 8]);
        lsch.extend_from_slice(&64u32.to_le_bytes());
        lsch.extend_from_slice(&(8u32 << 16).to_le_bytes());
        lsch.extend(field(joaat("Mass"), 8, 0x21, 0));
        lsch.extend(field(joaat("Gears"), 12, 0x14, 0));
        lsch.extend(field(joaat("Model"), 16, 0x4a, 0));
        lsch.extend(field(joaat("Enabled"), 20, 0x01, 0));
        lsch.extend(field(joaat("Ratio"), 22, 0x20, 0));
        lsch.extend(field(joaat("Offset"), 32, 0x33, 0));
        lsch.extend(field(joaat("Curve"), 48, 0x52 | (7 << 16), 0));
        lsch.extend(field(ELEMENT_NAME, 0, 0x21, 0));

        let mut file = section(b"PSIN", &body);
        file.extend(section(b"LSCH", &lsch));
        file.extend(section(b"LENM", &[]));
        file
    }

    #[test]
    fn export_then_apply_is_identity_and_edits_patch_in_place() {
        names::add(["TestThing", "Mass", "Gears", "Model", "Enabled", "Ratio", "Offset", "Curve", "cheetah2"]);
        let file = sample();
        let xml = export_xml(&file, 0).expect("export");
        for want in [
            "type=\"TestThing\"",
            "n=\"Mass\" t=\"f32\"",
            "v=\"1630\"",
            "n=\"Gears\" t=\"s32\"",
            "v=\"-7\"",
            "s=\"cheetah2\"",
            "v=\"true\"",
            "n=\"Ratio\" t=\"f16\"",
            "v=\"1 2 3 0.5\"",
            "t=\"array\" count=\"2\" to=\"1+0x0\"",
            "v=\"0.25\"",
            "v=\"bits:0x7fc00000\"",
        ] {
            assert!(xml.contains(want), "{want} missing in\n{xml}");
        }
        let same = apply_xml(&file, &xml).expect("apply unchanged");
        assert!(same.changes.is_empty(), "{:?}", same.changes);
        assert_eq!(same.data, file);

        let edited = xml.replace("v=\"1630\"", "v=\"1800.5\"").replace("v=\"0x", "v=\"coach\" was=\"0x").replace("v=\"1 2 3 0.5\"", "v=\"1 2 3 4\"");
        let done = apply_xml(&file, &edited).expect("apply edits");
        assert_eq!(done.changes.len(), 3, "{:?}", done.changes);
        assert_eq!(done.data.len(), file.len());
        let diff = file.iter().zip(&done.data).filter(|(a, b)| a != b).count();
        assert!(diff > 0 && diff <= 12);
        let again = export_xml(&done.data, 0).expect("re export");
        assert!(again.contains("v=\"1800.5\"") && again.contains("v=\"1 2 3 4\""));
        assert!(again.contains(&format!("v=\"0x{:08x}\"", joaat("coach"))));

        assert!(apply_xml(&file, &xml.replace("v=\"-7\"", "v=\"seven\"")).is_err());
        assert!(apply_xml(&file[..file.len() - 8], &xml).is_err());
    }

    #[test]
    fn half_floats_round_trip() {
        for h in [0u16, 0x8000, 0x3c00, 0xbc00, 0x3126, 0x4f80, 0x0001, 0x03ff, 0x7bff] {
            assert_eq!(f32_to_half(half_to_f32(h)), h, "0x{h:04x}");
        }
        assert_eq!(half_to_f32(0x4f80), 30.0);
    }

    // inflates the consecutive zlib streams of the vehicle physics file in data.rpf
    fn vehicle_physics() -> Option<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let levels = std::env::var("PERCYSTAR_LEVELS_RPF").ok()?;
        let path = std::path::Path::new(&levels).parent()?.join("data.rpf");
        let mut f = std::fs::File::open(path).ok()?;
        f.seek(SeekFrom::Start(0x1c35_b000)).ok()?;
        let mut raw = vec![0u8; 0x40000];
        let n = f.read(&mut raw).ok()?;
        raw.truncate(n);
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos + 2 <= raw.len() {
            let mut dec = flate2::Decompress::new(true);
            let mut chunk = Vec::with_capacity(1 << 17);
            if dec.decompress_vec(&raw[pos..], &mut chunk, flate2::FlushDecompress::Finish).is_err() || chunk.is_empty() {
                break;
            }
            out.extend_from_slice(&chunk);
            pos += dec.total_in() as usize;
            match (pos..(pos + 64).min(raw.len().saturating_sub(1))).find(|&k| raw[k] == 0x78 && matches!(raw[k + 1], 0x01 | 0x5e | 0x9c | 0xda)) {
                Some(next) if out.len() < 3_811_200 => pos = next,
                _ => break,
            }
        }
        (out.len() == 3_811_200).then_some(out)
    }

    // every psin file in a folder exports and applies back unchanged, PERCYSTAR_PSIN_DIR names the folder
    #[test]
    fn extracted_files_round_trip() {
        let Ok(dir) = std::env::var("PERCYSTAR_PSIN_DIR") else { return };
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        let (mut files, mut values, mut raw, mut dangling, mut skipped) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for entry in rd.flatten() {
            let Ok(file) = std::fs::read(entry.path()) else { continue };
            let Ok(xml) = export_xml(&file, 0) else {
                skipped += 1;
                continue;
            };
            let same = apply_xml(&file, &xml).unwrap_or_else(|e| panic!("{}: {e}", entry.path().display()));
            assert!(same.changes.is_empty(), "{}: {:?}", entry.path().display(), &same.changes[..same.changes.len().min(3)]);
            files += 1;
            values += xml.matches(" o=\"0x").count();
            raw += xml.matches(" t=\"raw:").count();
            dangling += xml.matches(" to=\"?").count();
        }
        eprintln!("{files} files round trip, {skipped} without schema, {values} editable values, {raw} raw fields, {dangling} unresolved pointers");
        assert!(dangling * 200 <= values.max(1), "too many pointers do not resolve");
    }

    #[test]
    fn real_vehicle_file_round_trips() {
        let Some(file) = vehicle_physics() else { return };
        eprintln!("vehicle physics file loaded, {} bytes", file.len());
        names::seed();
        if let Ok(dir) = std::env::var("PERCYSTAR_NAMES_DIR") {
            names::load_dir(std::path::Path::new(&dir));
        }
        let xml = export_xml(&file, 0).expect("export");
        let named = xml.matches(" n=\"").count() - xml.matches(" n=\"0x").count();
        eprintln!("{named} of {} field lines carry a name", xml.matches(" n=\"").count());
        assert!(xml.contains("v=\"1630\"") && xml.contains("v=\"16500\""));
        // the mass field carries the same name earlier games used
        let mass = xml.lines().find(|l| l.contains("n=\"fMass\"") && l.contains("v=\"1630\"")).expect("fMass 1630 line");
        assert!(mass.contains("t=\"f32\""), "{mass}");
        if let Some(at) = xml.find(mass) {
            eprintln!("{}", xml[at.saturating_sub(40)..].lines().skip(1).take(30).collect::<Vec<_>>().join("
"));
        }
        let same = apply_xml(&file, &xml).expect("apply unchanged");
        assert!(same.changes.is_empty(), "{:?}", &same.changes[..same.changes.len().min(5)]);
        assert!(same.data == file);
        let cut = export_xml(&file, 20_000).expect("preview");
        assert!(cut.len() < 40_000 && cut.ends_with("</psin>\n"));
    }
}
