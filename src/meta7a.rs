//! Minimal reader for RAGE 9 self-describing `0DRPz` (resource type 0x7a)
//! metadata. Vehicle configuration tables in `packs.rpf` use this format.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Member {
    pub hash: u32,
    pub offset: u32,
    pub ty: u8,
    pub reference: u16,
    pub extra: u32,
}

#[derive(Clone, Debug)]
pub struct StructDef {
    pub size: u32,
    pub members: Vec<Member>,
}

pub struct Meta7a<'a> {
    data: &'a [u8],
    pub structs: HashMap<u32, StructDef>,
    /// Struct hash, byte size and system-page offset.
    blocks: Vec<(u32, u32, usize)>,
}

fn u16_at(data: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(data.get(offset..offset + 2)?.try_into().ok()?))
}

fn u32_at(data: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(data.get(offset..offset + 4)?.try_into().ok()?))
}

fn u64_at(data: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(data.get(offset..offset + 8)?.try_into().ok()?))
}

fn system_offset(pointer: u64) -> Option<usize> {
    ((pointer >> 28) == 5).then_some((pointer & 0x0fff_ffff) as usize)
}

/// Clip dictionaries are also `0DRPz`, but their enum count at `+0x4A` is zero.
/// Vehicle and other schema resources have at least one enum definition.
pub fn has_schema_enums(data: &[u8]) -> bool {
    data.len() >= 0x50
        && data.get(0x10..0x15) == Some(b"0DRPz")
        && u16_at(data, 0x4a).is_some_and(|count| count > 0)
}

impl<'a> Meta7a<'a> {
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        if data.len() < 0x50 || data.get(0x10..0x15) != Some(b"0DRPz") {
            return None;
        }
        let struct_count = u16_at(data, 0x48)? as usize;
        let block_count = u16_at(data, 0x4c)? as usize;
        let struct_table = system_offset(u64_at(data, 0x28)?)?;
        let block_table = system_offset(u64_at(data, 0x38)?)?;
        if struct_table.checked_add(struct_count.checked_mul(32)?)? > data.len()
            || block_table.checked_add(block_count.checked_mul(16)?)? > data.len()
        {
            return None;
        }

        let mut structs = HashMap::with_capacity(struct_count);
        for index in 0..struct_count {
            let offset = struct_table + index * 32;
            let hash = u32_at(data, offset)?;
            let member_table = system_offset(u64_at(data, offset + 0x10)?)?;
            let size = u32_at(data, offset + 0x18)?;
            let member_count = u16_at(data, offset + 0x1e)? as usize;
            if member_table.checked_add(member_count.checked_mul(16)?)? > data.len() {
                return None;
            }
            let mut members = Vec::with_capacity(member_count);
            for member_index in 0..member_count {
                let member = member_table + member_index * 16;
                members.push(Member {
                    hash: u32_at(data, member)?,
                    offset: u32_at(data, member + 4)?,
                    ty: *data.get(member + 8)?,
                    reference: u16_at(data, member + 10)?,
                    extra: u32_at(data, member + 12)?,
                });
            }
            structs.insert(hash, StructDef { size, members });
        }

        let mut blocks = Vec::with_capacity(block_count);
        for index in 0..block_count {
            let offset = block_table + index * 16;
            let hash = u32_at(data, offset)?;
            let size = u32_at(data, offset + 4)?;
            let base = system_offset(u64_at(data, offset + 8)?).unwrap_or(usize::MAX);
            blocks.push((hash, size, base));
        }
        Some(Self { data, structs, blocks })
    }

    pub fn objects(&self, hash: u32) -> Vec<usize> {
        let Some(definition) = self.structs.get(&hash) else { return Vec::new() };
        let stride = definition.size as usize;
        if stride == 0 {
            return Vec::new();
        }
        let mut objects = Vec::new();
        for &(block_hash, byte_size, base) in &self.blocks {
            if block_hash != hash || base == usize::MAX {
                continue;
            }
            for index in 0..byte_size as usize / stride {
                let offset = base + index * stride;
                if offset + stride <= self.data.len() {
                    objects.push(offset);
                }
            }
        }
        objects
    }

    pub fn member(&self, struct_hash: u32, member_hash: u32) -> Option<&Member> {
        self.structs.get(&struct_hash)?.members.iter().find(|member| member.hash == member_hash)
    }

    pub fn object_u32(&self, struct_hash: u32, object: usize, member_hash: u32) -> Option<u32> {
        let member = self.member(struct_hash, member_hash)?;
        match member.ty {
            0x10 | 0x11 | 0x12 | 0x13 | 0x14 | 0x15 | 0x4a | 0x60 => {
                u32_at(self.data, object.checked_add(member.offset as usize)?)
            }
            _ => None,
        }
    }

    fn pointer(&self, value: u64) -> Option<(u32, usize)> {
        if let Some(offset) = system_offset(value).filter(|&offset| offset < self.data.len()) {
            return Some((0, offset));
        }
        let block_index = (value & 0xfff) as usize;
        let byte_offset = (value >> 20) as usize;
        let &(hash, size, base) = self.blocks.get(block_index.checked_sub(1)?)?;
        (base != usize::MAX && byte_offset < size as usize).then_some((hash, base + byte_offset))
    }

    pub fn object_u32_array(&self, struct_hash: u32, object: usize, member_hash: u32) -> Vec<u32> {
        let Some(definition) = self.structs.get(&struct_hash) else { return Vec::new() };
        let Some(member) = definition.members.iter().find(|member| member.hash == member_hash) else {
            return Vec::new();
        };
        let field = object + member.offset as usize;
        let (count, pointer_offset) = match member.ty {
            // Meta array: pointer at +0, count at +8.
            0x52 => (
                u16_at(self.data, field + 8).map(usize::from).unwrap_or(0),
                field,
            ),
            // PSO-style array used by packs.rpf vehicle configs: count, capacity, pointer.
            0x5c => (
                u32_at(self.data, field).map(|count| count as usize).unwrap_or(0),
                field + 8,
            ),
            _ => return Vec::new(),
        };
        if count == 0 || count > 100_000 {
            return Vec::new();
        }
        let Some(element) = definition.members.get(member.reference as usize) else { return Vec::new() };
        let stride = match element.ty {
            0x10 | 0x11 | 0x60 => 1,
            0x12 | 0x13 => 2,
            0x14 | 0x15 | 0x4a => 4,
            _ => return Vec::new(),
        };
        let Some((_, base)) = u64_at(self.data, pointer_offset).and_then(|pointer| self.pointer(pointer)) else {
            return Vec::new();
        };
        if base.checked_add(count.saturating_mul(stride)).is_none_or(|end| end > self.data.len()) {
            return Vec::new();
        }
        (0..count)
            .filter_map(|index| match stride {
                1 => self.data.get(base + index).copied().map(u32::from),
                2 => u16_at(self.data, base + index * 2).map(u32::from),
                4 => u32_at(self.data, base + index * 4),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::has_schema_enums;

    #[test]
    fn schema_enums_distinguish_vehicle_meta_from_clip_dictionaries() {
        let mut meta = vec![0u8; 0x50];
        meta[0x10..0x15].copy_from_slice(b"0DRPz");
        meta[0x4a..0x4c].copy_from_slice(&19u16.to_le_bytes());
        assert!(has_schema_enums(&meta));

        let mut clip = meta.clone();
        clip[0x4a..0x4c].copy_from_slice(&0u16.to_le_bytes());
        assert!(!has_schema_enums(&clip));
    }
}

