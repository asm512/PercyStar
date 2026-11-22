//! Authored fixed vehicle configurations from `packs.rpf` 0x7a metadata.

use crate::meta7a::Meta7a;
use std::collections::{HashMap, HashSet};

pub const CONFIG_STRUCT: u32 = 0x5101_9114;
const OWNER_FIELD: u32 = 0xfb1f_80ff;
const COLOUR_FIELD: u32 = 0x5bdc_13e3;
const PARTS_FIELD: u32 = 0xe89b_8540;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct VehicleConfiguration {
    pub colour_argb: Option<u32>,
    pub parts: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedConfigurationParts {
    pub entries: Vec<usize>,
    pub unresolved: Vec<u32>,
    pub needs_roofbar_mount: bool,
}

pub type VehicleConfigurationDb = HashMap<u32, Vec<VehicleConfiguration>>;

pub fn parse_configurations(data: &[u8]) -> VehicleConfigurationDb {
    let Some(meta) = Meta7a::parse(data) else { return HashMap::new() };
    if !meta.structs.contains_key(&CONFIG_STRUCT) {
        return HashMap::new();
    }
    let mut database: VehicleConfigurationDb = HashMap::new();
    let mut seen = HashSet::new();
    for object in meta.objects(CONFIG_STRUCT) {
        let Some(owner) = meta.object_u32(CONFIG_STRUCT, object, OWNER_FIELD).filter(|&hash| hash != 0) else {
            continue;
        };
        let configuration = VehicleConfiguration {
            colour_argb: meta.object_u32(CONFIG_STRUCT, object, COLOUR_FIELD),
            parts: meta.object_u32_array(CONFIG_STRUCT, object, PARTS_FIELD),
        };
        if seen.insert((owner, configuration.clone())) {
            database.entry(owner).or_default().push(configuration);
        }
    }
    database
}

pub fn merge_configurations(target: &mut VehicleConfigurationDb, source: VehicleConfigurationDb) {
    for (model, configurations) in source {
        let destination = target.entry(model).or_default();
        let mut seen: HashSet<VehicleConfiguration> = destination.iter().cloned().collect();
        for configuration in configurations {
            if seen.insert(configuration.clone()) {
                destination.push(configuration);
            }
        }
    }
}

pub fn argb_to_linear_rgb(argb: u32) -> [f32; 3] {
    [
        ((argb >> 16) & 0xff) as f32 / 255.0,
        ((argb >> 8) & 0xff) as f32 / 255.0,
        (argb & 0xff) as f32 / 255.0,
    ]
}

/// Prefer a matching detail LOD when indexing scanned fragment names by JOAAT.
pub fn prefer_lod_entry(
    current: Option<(usize, bool)>,
    candidate: usize,
    detail_match: bool,
) -> (usize, bool) {
    match current {
        Some(slot) if !(detail_match && !slot.1) => slot,
        _ => (candidate, detail_match),
    }
}

/// Resolve authored part hashes to catalog entries and flag shared roofbars that need a mount.
pub fn resolve_configuration_parts(
    model_hash: u32,
    part_hashes: &[u32],
    entries_by_hash: &HashMap<u32, (usize, bool)>,
    entry_names: &HashMap<usize, String>,
) -> ResolvedConfigurationParts {
    let mut entries = Vec::new();
    let mut unresolved = Vec::new();
    let mut needs_roofbar_mount = false;
    for &part_hash in part_hashes {
        if part_hash == model_hash {
            continue;
        }
        if let Some(&(entry, _)) = entries_by_hash.get(&part_hash) {
            if !entries.contains(&entry) {
                if entry_names
                    .get(&entry)
                    .is_some_and(|name| name.to_ascii_lowercase().starts_with("cmn_siren_roofbar1_"))
                {
                    needs_roofbar_mount = true;
                }
                entries.push(entry);
            }
        } else {
            unresolved.push(part_hash);
        }
    }
    ResolvedConfigurationParts {
        entries,
        unresolved,
        needs_roofbar_mount,
    }
}

/// Insert the vehicle-specific roofbar mount ahead of the shared lightbar when needed.
pub fn with_implicit_roofbar_mount(mut entries: Vec<usize>, mount: Option<usize>, needs_mount: bool) -> Vec<usize> {
    if needs_mount {
        if let Some(mount) = mount {
            if !entries.contains(&mount) {
                entries.insert(0, mount);
            }
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u16(data: &mut [u8], offset: usize, value: u16) {
        data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(data: &mut [u8], offset: usize, value: u32) {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(data: &mut [u8], offset: usize, value: u64) {
        data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn sys(offset: usize) -> u64 {
        0x5000_0000 | offset as u64
    }

    #[test]
    fn parses_fixed_configuration_and_part_array() {
        let mut data = vec![0u8; 0x340];
        data[0x10..0x15].copy_from_slice(b"0DRPz");
        put_u64(&mut data, 0x28, sys(0x80));
        put_u64(&mut data, 0x38, sys(0x180));
        put_u16(&mut data, 0x48, 1);
        put_u16(&mut data, 0x4c, 1);

        put_u32(&mut data, 0x80, CONFIG_STRUCT);
        put_u64(&mut data, 0x90, sys(0xc0));
        put_u32(&mut data, 0x98, 0x20);
        put_u16(&mut data, 0x9e, 4);
        // packs.rpf vehicle configs use PSO-style 0x5c part arrays.
        for (index, (hash, offset, ty, reference)) in [
            (OWNER_FIELD, 0, 0x4a, 0),
            (COLOUR_FIELD, 4, 0x15, 0),
            (PARTS_FIELD, 8, 0x5c, 3),
            (0x100, 0, 0x4a, 0),
        ]
        .into_iter()
        .enumerate()
        {
            let member = 0xc0 + index * 16;
            put_u32(&mut data, member, hash);
            put_u32(&mut data, member + 4, offset);
            data[member + 8] = ty;
            put_u16(&mut data, member + 10, reference);
        }

        put_u32(&mut data, 0x180, CONFIG_STRUCT);
        put_u32(&mut data, 0x184, 0x20);
        put_u64(&mut data, 0x188, sys(0x200));
        put_u32(&mut data, 0x200, 0x33f2_dea5);
        put_u32(&mut data, 0x204, 0xff12_3456);
        put_u32(&mut data, 0x208, 3); // count
        put_u32(&mut data, 0x20c, 3); // capacity
        put_u64(&mut data, 0x210, sys(0x300));
        put_u32(&mut data, 0x300, 0x1111_1111);
        put_u32(&mut data, 0x304, 0x2222_2222);
        put_u32(&mut data, 0x308, 0x3333_3333);

        let database = parse_configurations(&data);
        let configurations = database.get(&0x33f2_dea5).expect("stanierg2 configuration");
        assert_eq!(configurations.len(), 1);
        assert_eq!(configurations[0].colour_argb, Some(0xff12_3456));
        assert_eq!(configurations[0].parts, [0x1111_1111, 0x2222_2222, 0x3333_3333]);
    }

    #[test]
    fn merging_deduplicates_repeated_pack_rows() {
        let configuration = VehicleConfiguration {
            colour_argb: Some(0xff00_0000),
            parts: vec![1, 2, 3],
        };
        let mut target = HashMap::from([(7, vec![configuration.clone()])]);
        merge_configurations(&mut target, HashMap::from([(7, vec![configuration])]));
        assert_eq!(target[&7].len(), 1);
    }

    #[test]
    fn prefer_matching_lod_when_indexing_hashes() {
        assert_eq!(prefer_lod_entry(None, 10, false), (10, false));
        assert_eq!(prefer_lod_entry(Some((10, false)), 11, true), (11, true));
        assert_eq!(prefer_lod_entry(Some((11, true)), 12, false), (11, true));
    }

    #[test]
    fn resolves_parts_and_inserts_implicit_roofbar_mount() {
        let entries_by_hash = HashMap::from([(0xaaaa, (3usize, true)), (0xbbbb, (7usize, true))]);
        let entry_names = HashMap::from([
            (3usize, "rim_serv911_02_hi.#ft".into()),
            (7usize, "cmn_siren_roofbar1_02_hi.#ft".into()),
        ]);
        let resolved = resolve_configuration_parts(0x1111, &[0x1111, 0xaaaa, 0xbbbb, 0xcccc], &entries_by_hash, &entry_names);
        assert_eq!(resolved.entries, [3, 7]);
        assert_eq!(resolved.unresolved, [0xcccc]);
        assert!(resolved.needs_roofbar_mount);
        assert_eq!(
            with_implicit_roofbar_mount(resolved.entries, Some(42), true),
            [42, 3, 7]
        );
    }
}

