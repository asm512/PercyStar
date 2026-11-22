use flate2::read::DeflateDecoder;
use std::fmt::Write as _;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

// largest shader blob accepted after inflating
const MAX_SHADER_BYTES: usize = 64 << 20;

const MAGIC: &[u8; 4] = b"SGD5";
const COMPRESSED_MAGIC: u32 = 0x000D_DCE7;
const STAGES: [(&str, &str); 6] = [
    ("VS", "vs"),
    ("PS", "ps"),
    ("GS", "gs"),
    ("DS", "ds"),
    ("HS", "hs"),
    ("CS", "cs"),
];

#[derive(Clone, Debug)]
pub struct Sgd5Shader {
    pub stage: &'static str,
    pub index: u32,
    pub name: String,
    pub wave_size: u8,
    pub compressed_size: u32,
    pub uncompressed_size: u32,
    pub sga_hash: u64,
    pub register_count: u16,
    pub cbuffer_count: u16,
    pub texture_count: u16,
    pub binary_offset: usize,
    pub reflection_records: Vec<Sgd5ReflectionRecord>,
}
#[derive(Clone, Debug)]
pub struct Sgd5DescriptorEntry {
    pub name_hash: u32,
    pub kind0: u8,
    pub kind1: u8,
    pub index: u16,
}

#[derive(Clone, Debug)]
pub struct Sgd5Descriptor {
    pub hash: u64,
    pub unknown: [u8; 14],
    pub entries: Vec<Sgd5DescriptorEntry>,
}

#[derive(Clone, Debug)]
pub struct Sgd5ReflectionRecord {
    pub resource_type: u16,
    pub slot: u16,
    pub flags: u16,
    pub reserved: u16,
    pub name_offset: u32,
    pub unknown: u32,
    pub name_hash: u32,
    pub name_hash_duplicate: u32,
    pub name: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Sgd5TextureEntry {
    pub name_hash: u32,
    pub resource_type: u16,
    pub index: u16,
    pub name: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Sgd5SharedReflection {
    pub size: u32,
    pub referenced_count: u16,
    pub extra_count: u16,
    pub reserved: u16,
    pub records: Vec<Sgd5ReflectionRecord>,
}

#[derive(Clone, Debug)]
pub struct Sgd5ReflectionTable {
    pub size: u32,
    pub resource_count: u16,
    pub cbuffer_member_count: u16,
    pub texture_count: u16,
    pub resources: Vec<Sgd5ReflectionRecord>,
    pub cbuffer_members: Vec<Sgd5ReflectionRecord>,
    pub textures: Vec<Sgd5TextureEntry>,
    pub strings: Vec<String>,
    pub defaults: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Sgd5SecondaryReflection {
    pub size: u32,
    pub resource_count: u16,
    pub cbuffer_member_count: u16,
    pub texture_count: u16,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Sgd5PropertyBinding {
    pub name_hash: u32,
    pub tag: u32,
    pub name: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Sgd5PropertyEntry {
    pub offset: usize,
    pub size: u32,
    pub hash: u64,
    pub flags1: u32,
    pub flags2: u32,
    pub bindings: Vec<Sgd5PropertyBinding>,
    pub sentinel: bool,
}

#[derive(Clone, Debug)]
pub struct Sgd5Effect {
    pub name: String,
    pub data_buffer_size: u32,
    pub stage_indices: [Vec<u32>; 6],
    pub payload_offset: usize,
    pub reflection_offset: usize,
    pub payload_end: usize,
    pub reflection: Option<Sgd5ReflectionTable>,
    pub secondary_reflection: Option<Sgd5SecondaryReflection>,
    pub property_offset: Option<usize>,
    pub property_count: u16,
    pub property_flags: u16,
    pub properties: Vec<Sgd5PropertyEntry>,
    pub strings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Sgd5File {
    pub data: Vec<u8>,
    pub group_hash: u64,
    pub descriptors: Vec<Sgd5Descriptor>,
    pub shared_reflection: Sgd5SharedReflection,
    pub shaders: Vec<Sgd5Shader>,
    pub declared_effect_count: u32,
    pub effect_count: u32,
    pub trailer_flags: u32,
    pub trailer_reserved: [u8; 6],
    pub effects: Vec<Sgd5Effect>,
    pub hash_names: std::collections::HashMap<u32, String>,
    pub effect_shader_map: std::collections::HashMap<(u8, u32), Vec<usize>>,
}

impl Sgd5File {
    pub fn resolve_name(&self, hash: u32) -> Option<&str> {
        self.hash_names.get(&hash).map(String::as_str)
    }

    pub fn effect_names_for_shader(&self, stage: &str, index: u32) -> Vec<&str> {
        let Some(stage_index) = STAGES.iter().position(|(name, _)| *name == stage) else {
            return Vec::new();
        };
        self.effect_shader_map
            .get(&(stage_index as u8, index))
            .map(|effect_indices| {
                effect_indices
                    .iter()
                    .filter_map(|effect_index| self.effects.get(*effect_index))
                    .map(|effect| effect.name.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[derive(Debug)]
pub struct Sgd5Extraction {
    pub files: Vec<Sgd5File>,
    pub extracted_count: usize,
    pub manifests: Vec<PathBuf>,
}

#[derive(Debug)]
pub struct Sgd5Error(String);

impl std::fmt::Display for Sgd5Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Sgd5Error {}

fn rd16(data: &[u8], offset: usize) -> Result<u16, Sgd5Error> {
    data.get(offset..offset + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .ok_or_else(|| Sgd5Error(format!("truncated u16 at 0x{offset:x}")))
}

fn rd32(data: &[u8], offset: usize) -> Result<u32, Sgd5Error> {
    data.get(offset..offset + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| Sgd5Error(format!("truncated u32 at 0x{offset:x}")))
}

fn rd64(data: &[u8], offset: usize) -> Result<u64, Sgd5Error> {
    data.get(offset..offset + 8)
        .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .ok_or_else(|| Sgd5Error(format!("truncated u64 at 0x{offset:x}")))
}

fn read_string(data: &[u8], offset: usize, length: usize) -> Result<String, Sgd5Error> {
    let bytes = data
        .get(offset..offset + length)
        .ok_or_else(|| Sgd5Error(format!("truncated shader name at 0x{offset:x}")))?;
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
}

pub fn joaat_lower(name: &str) -> u32 {
    let mut hash = 0u32;
    for byte in name.as_bytes().iter().map(|byte| byte.to_ascii_lowercase()) {
        hash = hash.wrapping_add(byte as u32);
        hash = hash.wrapping_add(hash << 10);
        hash ^= hash >> 6;
    }
    hash = hash.wrapping_add(hash << 3);
    hash ^= hash >> 11;
    hash = hash.wrapping_add(hash << 15);
    hash
}

fn is_ascii_identifier(name: &str) -> bool {
    let mut characters = name.chars();
    match characters.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

fn parse_nul_pool(data: &[u8], offset: usize, length: usize) -> Result<Vec<String>, Sgd5Error> {
    let pool = data
        .get(offset..offset + length)
        .ok_or_else(|| Sgd5Error(format!("truncated string pool at 0x{offset:x}")))?;
    if pool.first() != Some(&0) {
        return Err(Sgd5Error(format!(
            "string pool at 0x{offset:x} does not start with NUL"
        )));
    }
    let mut strings = Vec::new();
    for token in pool[1..].split(|&byte| byte == 0) {
        if token.is_empty() {
            continue;
        }
        strings.push(String::from_utf8_lossy(token).into_owned());
    }
    Ok(strings)
}

fn read_reflection_record(data: &[u8], offset: usize) -> Result<Sgd5ReflectionRecord, Sgd5Error> {
    Ok(Sgd5ReflectionRecord {
        resource_type: rd16(data, offset)?,
        slot: rd16(data, offset + 2)?,
        flags: rd16(data, offset + 4)?,
        reserved: rd16(data, offset + 6)?,
        name_offset: rd32(data, offset + 8)?,
        unknown: rd32(data, offset + 12)?,
        name_hash: rd32(data, offset + 16)?,
        name_hash_duplicate: rd32(data, offset + 20)?,
        name: None,
    })
}

fn resolve_hashed_names(strings: &[String]) -> std::collections::HashMap<u32, String> {
    let mut names = std::collections::HashMap::with_capacity(strings.len() * 2);
    for name in strings {
        names.insert(joaat_lower(name), name.clone());
        if !name.starts_with('_') {
            let sampler_name = format!("_{name}");
            names.insert(joaat_lower(&sampler_name), sampler_name);
        }
    }
    names
}

fn parse_container_header(
    data: &[u8],
) -> Result<(u64, Vec<Sgd5Descriptor>, Sgd5SharedReflection, usize), Sgd5Error> {
    let group_hash = rd64(data, 4)?;
    let descriptor_count = rd32(data, 12)? as usize;
    if descriptor_count > 10_000 {
        return Err(Sgd5Error(format!(
            "invalid SGD5 descriptor count {descriptor_count}"
        )));
    }

    let mut cursor = 16usize;
    let mut descriptors = Vec::with_capacity(descriptor_count);
    for descriptor_index in 0..descriptor_count {
        let descriptor_start = cursor;
        let size = rd32(data, cursor)? as usize;
        cursor += 4;
        let hash = rd64(data, cursor)?;
        cursor += 8;
        let entry_count = rd16(data, cursor)? as usize;
        cursor += 2;
        let unknown = data
            .get(cursor..cursor + 14)
            .and_then(|bytes| <[u8; 14]>::try_from(bytes).ok())
            .ok_or_else(|| Sgd5Error(format!("descriptor {descriptor_index}: truncated header")))?;
        cursor += 14;
        if size != 24 + entry_count * 8 {
            return Err(Sgd5Error(format!(
                "descriptor {descriptor_index}: size {size} does not match {entry_count} entries"
            )));
        }

        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let kind0 = *data.get(cursor + 4).ok_or_else(|| {
                Sgd5Error(format!("descriptor {descriptor_index}: truncated entry"))
            })?;
            let kind1 = *data.get(cursor + 5).ok_or_else(|| {
                Sgd5Error(format!("descriptor {descriptor_index}: truncated entry"))
            })?;
            entries.push(Sgd5DescriptorEntry {
                name_hash: rd32(data, cursor)?,
                kind0,
                kind1,
                index: rd16(data, cursor + 6)?,
            });
            cursor += 8;
        }
        if cursor != descriptor_start + 4 + size {
            return Err(Sgd5Error(format!(
                "descriptor {descriptor_index}: decoded size mismatch"
            )));
        }
        descriptors.push(Sgd5Descriptor {
            hash,
            unknown,
            entries,
        });
    }

    let reflection_size = rd32(data, cursor)? as usize;
    cursor += 4;
    let reflection_start = cursor;
    let referenced_count = rd16(data, cursor)?;
    let extra_count = rd16(data, cursor + 2)?;
    let reserved = rd16(data, cursor + 4)?;
    let size_copy = rd16(data, cursor + 6)?;
    let record_count = referenced_count as usize + extra_count as usize;
    if size_copy as usize != reflection_size || reflection_size != 8 + record_count * 24 + 1 {
        return Err(Sgd5Error(format!(
            "shared reflection size {reflection_size} does not match {referenced_count} referenced and {extra_count} extra records"
        )));
    }
    cursor += 8;
    let mut records = Vec::with_capacity(record_count);
    for _ in 0..record_count {
        records.push(read_reflection_record(data, cursor)?);
        cursor += 24;
    }
    let arrays_offset = reflection_start + reflection_size;
    if arrays_offset > data.len() {
        return Err(Sgd5Error("shared reflection table is truncated".into()));
    }

    Ok((
        group_hash,
        descriptors,
        Sgd5SharedReflection {
            size: reflection_size as u32,
            referenced_count,
            extra_count,
            reserved,
            records,
        },
        arrays_offset,
    ))
}

fn parse_shader(
    data: &[u8],
    stage: &'static str,
    index: u32,
    offset: usize,
) -> Result<(Sgd5Shader, usize), Sgd5Error> {
    let name_length = *data
        .get(offset)
        .ok_or_else(|| Sgd5Error(format!("truncated shader at 0x{offset:x}")))?;
    let name = read_string(data, offset + 1, name_length as usize)?;
    let mut cursor = offset + 1 + name_length as usize;
    if name.is_empty() {
        return Err(Sgd5Error(format!("empty shader name at 0x{offset:x}")));
    }

    let wave_size = *data
        .get(cursor)
        .ok_or_else(|| Sgd5Error(format!("{name}: truncated wave size")))?;
    let compressed_size = rd32(data, cursor + 1)?;
    let binary_offset = cursor + 5;
    cursor = binary_offset + compressed_size as usize;
    let sga_hash = rd64(data, cursor)?;
    cursor += 8;

    let root_signature_end = cursor
        .checked_add(144)
        .ok_or_else(|| Sgd5Error(format!("{name}: root-signature offset overflow")))?;
    if root_signature_end > data.len() {
        return Err(Sgd5Error(format!("{name}: truncated root signature")));
    }
    cursor = root_signature_end;

    let block_size = rd32(data, cursor)?;
    cursor += 4;
    let block_start = cursor;
    let register_count = rd16(data, cursor)?;
    let cbuffer_count = rd16(data, cursor + 2)?;
    let texture_count = rd16(data, cursor + 4)?;
    let block_size_copy = rd16(data, cursor + 6)?;
    let expected_block_size = 8usize
        .checked_add(
            24usize
                .checked_mul(register_count as usize + cbuffer_count as usize)
                .ok_or_else(|| Sgd5Error(format!("{name}: reflection count overflow")))?,
        )
        .and_then(|size| size.checked_add(1))
        .ok_or_else(|| Sgd5Error(format!("{name}: block size overflow")))?;
    if block_size as usize != expected_block_size {
        return Err(Sgd5Error(format!(
            "{name}: block size {} does not match {} registers and {} cbuffers",
            block_size, register_count, cbuffer_count
        )));
    }
    if block_size_copy != block_size as u16 {
        return Err(Sgd5Error(format!(
            "{name}: duplicate block size {block_size_copy} does not match {block_size}"
        )));
    }

    let uncompressed_size = rd32(data, binary_offset + 8)?;
    if rd32(data, binary_offset)? != COMPRESSED_MAGIC {
        return Err(Sgd5Error(format!(
            "{name}: invalid compressed-container magic"
        )));
    }
    if compressed_size < 12 {
        return Err(Sgd5Error(format!(
            "{name}: compressed container is too short"
        )));
    }

    let reflection_count = register_count as usize + cbuffer_count as usize;
    let mut reflection_records = Vec::with_capacity(reflection_count);
    for record_index in 0..reflection_count {
        reflection_records.push(read_reflection_record(
            data,
            block_start + 8 + record_index * 24,
        )?);
    }

    cursor = block_start + block_size as usize;
    if cursor > data.len() {
        return Err(Sgd5Error(format!("{name}: truncated reflection block")));
    }

    Ok((
        Sgd5Shader {
            stage,
            index,
            name,
            wave_size,
            compressed_size,
            uncompressed_size,
            sga_hash,
            register_count,
            cbuffer_count,
            texture_count,
            binary_offset,
            reflection_records,
        },
        cursor,
    ))
}

struct Sgd5EffectHeader {
    name: String,
    data_buffer_size: u32,
    stage_indices: [Vec<u32>; 6],
    payload_offset: usize,
}

struct Sgd5EffectPreReflection {
    reflection_offset: usize,
}

fn checked_add_offset(offset: usize, size: usize, label: &str) -> Result<usize, Sgd5Error> {
    offset
        .checked_add(size)
        .ok_or_else(|| Sgd5Error(format!("SGD5 effect {label} offset overflow")))
}

fn parse_effect_pre_reflection(
    data: &[u8],
    payload_offset: usize,
) -> Result<Sgd5EffectPreReflection, Sgd5Error> {
    let base_end = checked_add_offset(payload_offset, 0x5c, "metadata base")?;
    if base_end > data.len() {
        return Err(Sgd5Error("truncated SGD5 effect metadata base".into()));
    }

    let extension_count = rd32(data, payload_offset + 0x50)? as usize;
    if extension_count > 4 {
        return Err(Sgd5Error(format!(
            "invalid SGD5 effect extension count {extension_count}"
        )));
    }

    let mut cursor = base_end;
    let mut tail_offset = payload_offset + 0x58;
    for extension_index in 0..extension_count {
        let extension_offset = cursor;
        cursor = checked_add_offset(cursor, 0x10, "extension")?;
        if cursor > data.len() {
            return Err(Sgd5Error(format!(
                "truncated SGD5 effect extension {extension_index}"
            )));
        }
        if rd32(data, extension_offset)? != 0 || rd32(data, extension_offset + 4)? != 0x0000_0101 {
            return Err(Sgd5Error(format!(
                "invalid SGD5 effect extension {extension_index} at 0x{extension_offset:x}"
            )));
        }
        tail_offset += 0x10;
    }

    let mut continuation_count = 0usize;
    while rd32(data, tail_offset)? >> 16 != 0 {
        cursor = checked_add_offset(cursor, 0x18, "continuation")?;
        tail_offset = checked_add_offset(tail_offset, 0x18, "continuation tail")?;
        if cursor > data.len() {
            return Err(Sgd5Error("truncated SGD5 effect continuation".into()));
        }
        continuation_count += 1;
        if continuation_count > 1_024 {
            return Err(Sgd5Error("too many SGD5 effect continuations".into()));
        }
    }

    let tail = rd32(data, tail_offset)?;
    if tail != 0 && tail & 0xff00 == 0 {
        let state_chunk_count = (tail & 0xff) as usize;
        let state_chunk_size = state_chunk_count
            .checked_mul(0x44)
            .ok_or_else(|| Sgd5Error("SGD5 effect state-chunk size overflow".into()))?;
        cursor = checked_add_offset(cursor, state_chunk_size, "state chunks")?;
        if cursor > data.len() {
            return Err(Sgd5Error("truncated SGD5 effect state chunks".into()));
        }
    }

    let read_counted_part =
        |cursor: &mut usize, entry_size: usize, label: &str| -> Result<usize, Sgd5Error> {
            if *cursor + 4 > data.len() {
                return Err(Sgd5Error(format!("truncated SGD5 effect {label} count")));
            }
            let count = rd32(data, *cursor)? as usize;
            *cursor += 4;
            let body_size = count
                .checked_mul(entry_size)
                .ok_or_else(|| Sgd5Error(format!("SGD5 effect {label} size overflow")))?;
            *cursor = checked_add_offset(*cursor, body_size, label)?;
            if *cursor > data.len() {
                return Err(Sgd5Error(format!("truncated SGD5 effect {label}")));
            }
            Ok(count)
        };

    read_counted_part(&mut cursor, 0x24, "part 1")?;
    read_counted_part(&mut cursor, 4, "part 2")?;
    read_counted_part(&mut cursor, 0x18, "part 3")?;
    read_counted_part(&mut cursor, 4, "part 4")?;

    if cursor + 4 > data.len() || rd32(data, cursor)? != 0x8000_0000 {
        return Err(Sgd5Error(format!(
            "missing SGD5 reflection sentinel at 0x{cursor:x}"
        )));
    }

    Ok(Sgd5EffectPreReflection {
        reflection_offset: cursor + 4,
    })
}

fn parse_effect_header(
    data: &[u8],
    offset: usize,
    stage_counts: &[u32; 6],
) -> Result<Sgd5EffectHeader, Sgd5Error> {
    let name_length = *data
        .get(offset)
        .ok_or_else(|| Sgd5Error(format!("truncated effect at 0x{offset:x}")))?
        as usize;
    if !(2..=128).contains(&name_length) {
        return Err(Sgd5Error(format!(
            "effect at 0x{offset:x}: invalid name length {name_length}"
        )));
    }
    let name_end = offset + name_length;
    if data.get(name_end) != Some(&0) {
        return Err(Sgd5Error(format!(
            "effect at 0x{offset:x}: missing name terminator"
        )));
    }
    let name_bytes = data
        .get(offset + 1..name_end)
        .ok_or_else(|| Sgd5Error(format!("truncated effect at 0x{offset:x}")))?;
    if name_bytes.contains(&0) {
        return Err(Sgd5Error(format!(
            "effect at 0x{offset:x}: embedded name terminator"
        )));
    }
    let name = String::from_utf8_lossy(name_bytes).into_owned();
    if !is_ascii_identifier(&name) {
        return Err(Sgd5Error(format!(
            "effect at 0x{offset:x}: invalid name {name}"
        )));
    }

    let mut cursor = name_end + 1;
    let data_buffer_size = rd32(data, cursor)?;
    cursor += 4;
    let mut stage_indices = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    for (stage, (indices, expected_count)) in stage_indices
        .iter_mut()
        .zip(stage_counts.iter())
        .enumerate()
    {
        let count = rd32(data, cursor)? as usize;
        cursor += 4;
        if count > *expected_count as usize {
            return Err(Sgd5Error(format!(
                "effect {name}: {stage} index count {count} exceeds {}",
                expected_count
            )));
        }
        indices.reserve(count);
        for _ in 0..count {
            let index = rd32(data, cursor)?;
            cursor += 4;
            if index >= *expected_count {
                return Err(Sgd5Error(format!(
                    "effect {name}: invalid {stage} shader index {index}"
                )));
            }
            indices.push(index);
        }
    }

    Ok(Sgd5EffectHeader {
        name,
        data_buffer_size,
        stage_indices,
        payload_offset: cursor,
    })
}

#[derive(Debug)]
struct Sgd5PropertySection {
    property_offset: usize,
    property_count: u16,
    property_flags: u16,
    entries: Vec<Sgd5PropertyEntry>,
    strings: Vec<String>,
    effect_end: usize,
}

fn parse_property_section(
    data: &[u8],
    property_offset: usize,
) -> Result<Sgd5PropertySection, Sgd5Error> {
    if property_offset + 4 > data.len() {
        return Err(Sgd5Error("truncated SGD5 property header".into()));
    }

    let property_count = rd16(data, property_offset)? as usize;
    let property_flags = rd16(data, property_offset + 2)?;
    let mut cursor = property_offset + 4;
    let mut entries = Vec::with_capacity(property_count);
    for index in 0..property_count {
        let blob_offset = cursor;
        let size_word = rd32(data, cursor)?;
        cursor += 4;

        if size_word == 0x8000_0000 {
            if cursor + 4 > data.len() {
                return Err(Sgd5Error(format!(
                    "SGD5 sentinel property {index} is truncated"
                )));
            }
            let hash = rd32(data, cursor)? as u64;
            cursor += 4;
            entries.push(Sgd5PropertyEntry {
                offset: blob_offset,
                size: size_word,
                hash,
                flags1: 0,
                flags2: 0,
                bindings: Vec::new(),
                sentinel: true,
            });
            continue;
        }

        let size = size_word as usize;
        if size < 24 {
            return Err(Sgd5Error(format!(
                "SGD5 property {index} has invalid size {size}"
            )));
        }
        let entry_start = cursor;
        let entry_end = entry_start
            .checked_add(size)
            .ok_or_else(|| Sgd5Error("SGD5 property size overflow".into()))?;
        if entry_end > data.len() {
            return Err(Sgd5Error(format!(
                "SGD5 property {index} is truncated at 0x{blob_offset:x}"
            )));
        }

        let count_word = rd32(data, entry_start + 8)?;
        let binding_count = (count_word & 0xff) as usize;
        if size != 24 + binding_count * 8 {
            return Err(Sgd5Error(format!(
                "SGD5 property {index} size {size} does not match {binding_count} bindings"
            )));
        }

        let hash = rd64(data, entry_start)?;
        let flags1 = count_word >> 8;
        let flags2 = rd32(data, entry_start + 12)?;
        let mut bindings = Vec::with_capacity(binding_count);
        for binding_index in 0..binding_count {
            let binding_offset = entry_start + 24 + binding_index * 8;
            bindings.push(Sgd5PropertyBinding {
                name_hash: rd32(data, binding_offset)?,
                tag: rd32(data, binding_offset + 4)?,
                name: None,
            });
        }
        entries.push(Sgd5PropertyEntry {
            offset: blob_offset,
            size: size_word,
            hash,
            flags1,
            flags2,
            bindings,
            sentinel: false,
        });
        cursor = entry_end;
    }

    let strings_size = rd32(data, cursor)? as usize;
    let strings_start = cursor + 4;
    let strings_end = strings_start
        .checked_add(strings_size)
        .ok_or_else(|| Sgd5Error("SGD5 string-pool size overflow".into()))?;
    if strings_size == 0 || strings_end > data.len() {
        return Err(Sgd5Error("invalid SGD5 string-pool size".into()));
    }
    let strings = parse_nul_pool(data, strings_start, strings_size)?;

    Ok(Sgd5PropertySection {
        property_offset,
        property_count: property_count as u16,
        property_flags,
        entries,
        strings,
        effect_end: strings_end,
    })
}

fn parse_reflection_table(
    data: &[u8],
    offset: usize,
) -> Result<(Sgd5ReflectionTable, usize), Sgd5Error> {
    let size = rd32(data, offset)? as usize;
    let table_end = offset
        .checked_add(4)
        .ok_or_else(|| Sgd5Error("SGD5 reflection-table offset overflow".into()))?
        .checked_add(size)
        .ok_or_else(|| Sgd5Error("SGD5 reflection-table size overflow".into()))?;
    if size < 8 || table_end > data.len() {
        return Err(Sgd5Error("invalid SGD5 reflection-table size".into()));
    }

    let resource_count = rd16(data, offset + 4)? as usize;
    let cbuffer_member_count = rd16(data, offset + 6)? as usize;
    let texture_count = rd16(data, offset + 8)? as usize;
    let size_copy = rd16(data, offset + 10)? as usize;
    if size_copy != size {
        return Err(Sgd5Error("SGD5 reflection-table size copy mismatch".into()));
    }

    let record_count = resource_count + cbuffer_member_count;
    let fixed_size = 8 + record_count * 24 + texture_count * 8 + texture_count * 4;
    if size < fixed_size {
        return Err(Sgd5Error("SGD5 reflection table is too small".into()));
    }
    let strings_size = size - fixed_size;
    let records_start = offset + 12;
    let textures_start = records_start + record_count * 24;
    let strings_start = textures_start + texture_count * 8;
    let defaults_start = strings_start + strings_size;
    if defaults_start + texture_count * 4 != table_end {
        return Err(Sgd5Error("SGD5 reflection-table extent mismatch".into()));
    }

    let mut resources = Vec::with_capacity(resource_count);
    for index in 0..resource_count {
        resources.push(read_reflection_record(data, records_start + index * 24)?);
    }
    let mut cbuffer_members = Vec::with_capacity(cbuffer_member_count);
    for index in 0..cbuffer_member_count {
        cbuffer_members.push(read_reflection_record(
            data,
            records_start + (resource_count + index) * 24,
        )?);
    }
    let mut textures = Vec::with_capacity(texture_count);
    for index in 0..texture_count {
        let entry_offset = textures_start + index * 8;
        textures.push(Sgd5TextureEntry {
            name_hash: rd32(data, entry_offset)?,
            resource_type: rd16(data, entry_offset + 4)?,
            index: rd16(data, entry_offset + 6)?,
            name: None,
        });
    }
    let strings = parse_nul_pool(data, strings_start, strings_size)?;
    let names = resolve_hashed_names(&strings);
    for record in resources.iter_mut().chain(cbuffer_members.iter_mut()) {
        record.name = names.get(&record.name_hash).cloned();
    }
    for texture in textures.iter_mut() {
        texture.name = names.get(&texture.name_hash).cloned();
    }
    let mut defaults = Vec::with_capacity(texture_count);
    for index in 0..texture_count {
        let default_offset = defaults_start + index * 4;
        defaults.push(f32::from_le_bytes([
            data[default_offset],
            data[default_offset + 1],
            data[default_offset + 2],
            data[default_offset + 3],
        ]));
    }

    Ok((
        Sgd5ReflectionTable {
            size: size as u32,
            resource_count: resource_count as u16,
            cbuffer_member_count: cbuffer_member_count as u16,
            texture_count: texture_count as u16,
            resources,
            cbuffer_members,
            textures,
            strings,
            defaults,
        },
        table_end,
    ))
}

fn parse_secondary_reflection_table(
    data: &[u8],
    offset: usize,
) -> Result<(Sgd5SecondaryReflection, usize), Sgd5Error> {
    let size = rd32(data, offset)? as usize;
    let table_end = offset
        .checked_add(4)
        .ok_or_else(|| Sgd5Error("SGD5 secondary-reflection offset overflow".into()))?
        .checked_add(size)
        .ok_or_else(|| Sgd5Error("SGD5 secondary-reflection size overflow".into()))?;
    if size < 8 || table_end > data.len() {
        return Err(Sgd5Error("invalid SGD5 secondary-reflection size".into()));
    }

    let resource_count = rd16(data, offset + 4)?;
    let cbuffer_member_count = rd16(data, offset + 6)?;
    let texture_count = rd16(data, offset + 8)?;
    let size_copy = rd16(data, offset + 10)? as usize;
    if size_copy != size {
        return Err(Sgd5Error(
            "SGD5 secondary-reflection size copy mismatch".into(),
        ));
    }

    Ok((
        Sgd5SecondaryReflection {
            size: size as u32,
            resource_count,
            cbuffer_member_count,
            texture_count,
            data: data[offset + 12..table_end].to_vec(),
        },
        table_end,
    ))
}

fn parse_effect(
    data: &[u8],
    start: usize,
    stage_counts: &[u32; 6],
) -> Result<(Sgd5Effect, usize), Sgd5Error> {
    let header = parse_effect_header(data, start, stage_counts)?;
    let pre_reflection = parse_effect_pre_reflection(data, header.payload_offset)?;
    let (reflection, secondary_offset) =
        parse_reflection_table(data, pre_reflection.reflection_offset)?;
    let (secondary_reflection, property_offset) =
        parse_secondary_reflection_table(data, secondary_offset)?;
    let section = parse_property_section(data, property_offset)?;
    if !is_effect_boundary(data, section.effect_end, stage_counts) {
        return Err(Sgd5Error(format!(
            "invalid SGD5 effect boundary at 0x{:x}",
            section.effect_end
        )));
    }

    Ok((
        Sgd5Effect {
            name: header.name,
            data_buffer_size: header.data_buffer_size,
            stage_indices: header.stage_indices,
            payload_offset: header.payload_offset,
            reflection_offset: pre_reflection.reflection_offset,
            payload_end: section.effect_end,
            reflection: Some(reflection),
            secondary_reflection: Some(secondary_reflection),
            property_offset: Some(section.property_offset),
            property_count: section.property_count,
            property_flags: section.property_flags,
            properties: section.entries,
            strings: section.strings,
        },
        section.effect_end,
    ))
}

fn is_effect_boundary(data: &[u8], offset: usize, stage_counts: &[u32; 6]) -> bool {
    if offset > data.len() {
        return false;
    }
    if offset == data.len() {
        return true;
    }
    if parse_effect_header(data, offset, stage_counts).is_ok() {
        return true;
    }
    if data.get(offset..offset + 4) == Some(MAGIC) {
        return true;
    }
    if is_stream_trailer(data, offset) {
        return true;
    }
    offset + 4 <= data.len()
        && (1..=10_000).contains(&rd32(data, offset).unwrap_or(0))
        && parse_effect_header(data, offset + 4, stage_counts).is_ok()
}

fn is_stream_trailer(data: &[u8], offset: usize) -> bool {
    if offset + 10 > data.len() {
        return false;
    }
    let flags = rd32(data, offset).unwrap_or(0);
    if !matches!(flags, 0 | 1) {
        return false;
    }
    if data[offset + 4..offset + 10].iter().any(|&byte| byte != 0) {
        return false;
    }

    true
}

#[derive(Debug)]
struct Sgd5EffectSection {
    effects: Vec<Sgd5Effect>,
    trailer_flags: u32,
    trailer_reserved: [u8; 6],
}

fn parse_effects_structured(
    data: &[u8],
    effect_region: usize,
    declared_effect_count: u32,
    stage_counts: &[u32; 6],
) -> Result<Sgd5EffectSection, Sgd5Error> {
    if declared_effect_count == 0 {
        return Err(Sgd5Error("SGD5 file declares no effects".into()));
    }

    let mut effects = Vec::with_capacity(declared_effect_count.min(10_000) as usize);
    let mut cursor = effect_region;
    while cursor < data.len() {
        if is_stream_trailer(data, cursor) {
            let trailer_flags = rd32(data, cursor)?;
            let mut trailer_reserved = [0u8; 6];
            trailer_reserved.copy_from_slice(&data[cursor + 4..cursor + 10]);
            return Ok(Sgd5EffectSection {
                effects,
                trailer_flags,
                trailer_reserved,
            });
        }
        if data.get(cursor..cursor + 4) == Some(MAGIC) {
            return Err(Sgd5Error(
                "SGD5 effect array ended without a stream trailer".into(),
            ));
        }

        let index = effects.len();
        let (effect, next) = parse_effect(data, cursor, stage_counts)
            .map_err(|error| Sgd5Error(format!("SGD5 effect {index} at 0x{cursor:x}: {error}")))?;
        effects.push(effect);
        cursor = next;
    }

    Err(Sgd5Error("SGD5 stream trailer is missing".into()))
}

fn build_hash_names(effects: &[Sgd5Effect]) -> std::collections::HashMap<u32, String> {
    let mut names = std::collections::HashMap::new();
    for effect in effects {
        for string in effect.strings.iter().chain(
            effect
                .reflection
                .iter()
                .flat_map(|table| table.strings.iter()),
        ) {
            names.insert(joaat_lower(string), string.clone());
            if !string.starts_with('_') {
                let sampler_name = format!("_{string}");
                names.insert(joaat_lower(&sampler_name), sampler_name);
            }
        }
    }
    names
}

fn build_effect_shader_map(
    effects: &[Sgd5Effect],
) -> std::collections::HashMap<(u8, u32), Vec<usize>> {
    let mut map = std::collections::HashMap::new();
    for (effect_index, effect) in effects.iter().enumerate() {
        for (stage_index, indices) in effect.stage_indices.iter().enumerate() {
            for shader_index in indices {
                map.entry((stage_index as u8, *shader_index))
                    .or_insert_with(Vec::new)
                    .push(effect_index);
            }
        }
    }
    map
}

pub fn load_sgd5(path: &Path) -> Result<Vec<Sgd5File>, Sgd5Error> {
    let data =
        fs::read(path).map_err(|error| Sgd5Error(format!("read {}: {error}", path.display())))?;
    parse_sgd5_streams(path, data)
}

fn parse_sgd5_data(path: &Path, data: Vec<u8>) -> Result<Sgd5File, Sgd5Error> {
    if data.len() < 0x20 || &data[..4] != MAGIC {
        return Err(Sgd5Error(format!(
            "{} is not an SGD5 shader-group file",
            path.display()
        )));
    }

    let (group_hash, descriptors, shared_reflection, mut offset) = parse_container_header(&data)?;
    let mut shaders = Vec::new();
    let mut stage_counts = [0u32; 6];
    for (stage_index, (stage, _)) in STAGES.iter().enumerate() {
        let count = rd32(&data, offset)?;
        offset += 4;
        if count > 100_000 {
            return Err(Sgd5Error(format!("invalid {stage} shader count {count}")));
        }
        stage_counts[stage_index] = count;
        for index in 0..count {
            let (shader, next_offset) = parse_shader(&data, stage, index, offset)?;
            shaders.push(shader);
            offset = next_offset;
        }
    }

    let declared_effect_count = rd32(&data, offset)?;
    let section =
        parse_effects_structured(&data, offset + 4, declared_effect_count, &stage_counts)?;
    let effects = section.effects;
    let hash_names = build_hash_names(&effects);
    let effect_shader_map = build_effect_shader_map(&effects);
    let mut effects = effects;
    for effect in effects.iter_mut() {
        for property in effect.properties.iter_mut() {
            for binding in property.bindings.iter_mut() {
                binding.name = hash_names.get(&binding.name_hash).cloned();
            }
        }
    }
    for shader in shaders.iter_mut() {
        for record in shader.reflection_records.iter_mut() {
            record.name = hash_names.get(&record.name_hash).cloned();
        }
    }
    Ok(Sgd5File {
        data,
        group_hash,
        descriptors,
        shared_reflection,
        shaders,
        declared_effect_count,
        effect_count: effects.len() as u32,
        trailer_flags: section.trailer_flags,
        trailer_reserved: section.trailer_reserved,
        effects,
        hash_names,
        effect_shader_map,
    })
}

fn find_sgd5_starts(data: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut offset = 0usize;
    while let Some(found) = data[offset..]
        .windows(MAGIC.len())
        .position(|window| window == MAGIC)
        .map(|position| offset + position)
    {
        starts.push(found);
        offset = found + 1;
    }
    starts
}

fn parse_sgd5_streams(path: &Path, data: Vec<u8>) -> Result<Vec<Sgd5File>, Sgd5Error> {
    let starts = find_sgd5_starts(&data);
    let mut files = Vec::new();
    for (stream_index, &start) in starts.iter().enumerate() {
        let end = starts.get(stream_index + 1).copied().unwrap_or(data.len());
        match parse_sgd5_data(path, data[start..end].to_vec()) {
            Ok(file) => files.push(file),
            Err(error) => {
                return Err(Sgd5Error(format!(
                    "SGD5 stream {stream_index} at 0x{start:x}: {error}"
                )))
            }
        }
    }
    if files.is_empty() {
        return Err(Sgd5Error("no valid SGD5 data".into()));
    }
    Ok(files)
}

pub fn decompress_shader(file: &Sgd5File, shader: &Sgd5Shader) -> Result<Vec<u8>, Sgd5Error> {
    let compressed = file
        .data
        .get(shader.binary_offset + 12..shader.binary_offset + shader.compressed_size as usize)
        .ok_or_else(|| Sgd5Error(format!("{}: truncated compressed payload", shader.name)))?;
    // the size field is a file value bound it and the inflate so a bomb cannot exhaust memory
    let expected = shader.uncompressed_size as usize;
    if expected > MAX_SHADER_BYTES {
        return Err(Sgd5Error(format!("{}: implausible uncompressed size {expected}", shader.name)));
    }
    let mut decoder = DeflateDecoder::new(compressed).take(expected as u64 + 1);
    let mut output = Vec::with_capacity(expected);
    decoder
        .read_to_end(&mut output)
        .map_err(|error| Sgd5Error(format!("{}: DEFLATE error: {error}", shader.name)))?;
    if output.len() != shader.uncompressed_size as usize {
        return Err(Sgd5Error(format!(
            "{}: decompressed {} bytes, expected {}",
            shader.name,
            output.len(),
            shader.uncompressed_size
        )));
    }
    if output.len() < 32 || &output[..4] != b"DXBC" {
        return Err(Sgd5Error(format!(
            "{}: decompressed payload is not DXBC",
            shader.name
        )));
    }
    Ok(output)
}

fn safe_name(name: &str) -> String {
    let mut output = String::with_capacity(name.len());
    for character in name.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.') {
            output.push(character);
        } else {
            output.push('_');
        }
    }
    let trimmed = output.trim_matches('_');
    if trimmed.is_empty() {
        "shader".to_string()
    } else {
        trimmed.to_string()
    }
}

fn extract_sgd5_file(
    file: &Sgd5File,
    output_dir: &Path,
    limit: Option<usize>,
    manifest_name: &str,
) -> Result<usize, Sgd5Error> {
    fs::create_dir_all(output_dir)
        .map_err(|error| Sgd5Error(format!("create {}: {error}", output_dir.display())))?;

    let selected = limit.map_or(&file.shaders[..], |limit| {
        &file.shaders[..limit.min(file.shaders.len())]
    });

    let mut manifest = String::from(
        "index,stage,name,wave_size,compressed_size,uncompressed_size,sga_hash,registers,cbuffers,textures,filename\n",
    );
    for shader in selected {
        let blob = decompress_shader(file, shader)?;
        let filename = format!(
            "{:05}_{}_{}.dxbc",
            shader.index,
            shader.stage.to_lowercase(),
            safe_name(&shader.name)
        );
        fs::write(output_dir.join(&filename), &blob)
            .map_err(|error| Sgd5Error(format!("write {filename}: {error}")))?;
        let _ = writeln!(
            manifest,
            "{},{},{},{},{},{},0x{:016x},{},{},{},{}",
            shader.index,
            shader.stage,
            shader.name,
            shader.wave_size,
            shader.compressed_size,
            shader.uncompressed_size,
            shader.sga_hash,
            shader.register_count,
            shader.cbuffer_count,
            shader.texture_count,
            filename
        );
    }
    fs::write(output_dir.join(manifest_name), manifest)
        .map_err(|error| Sgd5Error(format!("write {manifest_name}: {error}")))?;

    Ok(selected.len())
}

pub fn extract_sgd5(
    input: &Path,
    output_dir: &Path,
    limit: usize,
) -> Result<Sgd5Extraction, Sgd5Error> {
    let data =
        fs::read(input).map_err(|error| Sgd5Error(format!("read {}: {error}", input.display())))?;
    let files = parse_sgd5_streams(input, data)?;
    fs::create_dir_all(output_dir)
        .map_err(|error| Sgd5Error(format!("create {}: {error}", output_dir.display())))?;

    let mut remaining = if limit == 0 { None } else { Some(limit) };
    let mut extracted_count = 0usize;
    let mut manifests = Vec::new();
    for (stream_index, file) in files.iter().enumerate() {
        let single_output = if files.len() == 1 {
            output_dir.to_path_buf()
        } else {
            output_dir.join(format!("sgd5_{stream_index:03}"))
        };
        let manifest_name = "sgd5_index.csv";
        let extracted = extract_sgd5_file(file, &single_output, remaining, manifest_name)?;
        if let Some(remaining) = remaining.as_mut() {
            *remaining = remaining.saturating_sub(extracted);
        }
        extracted_count += extracted;
        manifests.push(single_output.join(manifest_name));
    }

    Ok(Sgd5Extraction {
        files,
        extracted_count,
        manifests,
    })
}
