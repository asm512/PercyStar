use crate::model3d::{Mesh3D, Texture2D};
use crate::worldmap::WorldMapData;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Icon, Window, WindowId};

mod smoke;
pub use smoke::run_gpu_world_smoke;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuInstance {
    pub model_0: [f32; 4],
    pub model_1: [f32; 4],
    pub model_2: [f32; 4],
    pub model_3: [f32; 4],
    pub color: [f32; 4],
}

impl GpuInstance {
    pub fn from_mat4(m: Mat4, color: [f32; 4]) -> Self {
        let cols = m.to_cols_array_2d();
        Self {
            model_0: cols[0],
            model_1: cols[1],
            model_2: cols[2],
            model_3: cols[3],
            color,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    light_vp: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sun_dir: [f32; 4],
    sky_zenith: [f32; 4],
    sky_horizon: [f32; 4],
    flags: [f32; 4], // x=rt_on, y=time_hours, z=map_min_x, w=map_span_x
    map_y: [f32; 4], // x=map_min_y, y=map_span_y, z=viewport_w, w=viewport_h
    hud: [f32; 4],   // x=load_progress, y=loading_active, z=elapsed_sec, w=fog_density
}

pub struct WorldBatchInput {
    pub mesh: Mesh3D,
    pub instances: Vec<(Vec3, Quat, Vec3, [f32; 4])>,
}

pub struct WorldLaunchConfig {
    pub terrain: Option<Arc<WorldMapData>>,
    pub batches: Vec<WorldBatchInput>,
    pub levels_rpf: Option<std::path::PathBuf>,
    pub map_count: usize,
    pub start_pos: [f32; 3],
}

// index list of one material group already in the width the gpu buffer uses
pub enum PreparedIndices {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl PreparedIndices {
    fn len(&self) -> usize {
        match self {
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
        }
    }

    fn format(&self) -> wgpu::IndexFormat {
        match self {
            Self::U16(_) => wgpu::IndexFormat::Uint16,
            Self::U32(_) => wgpu::IndexFormat::Uint32,
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Self::U16(v) => bytemuck::cast_slice(v),
            Self::U32(v) => bytemuck::cast_slice(v),
        }
    }
}

pub struct PreparedGeom {
    pub indices: PreparedIndices,
    // decoded texture shared with other materials and models, none when the mesh carries its own
    pub texture: Option<Arc<Texture2D>>,
}

// one run of placements in the same map cell
pub struct PreparedCell {
    pub inst_first: u32,
    pub inst_count: u32,
    pub bound_center: Vec3,
    pub bound_radius: f32,
    // radius of the largest placed copy in the cell
    pub obj_radius: f32,
}

// cpu side of one model built off the render thread, only buffers and bind groups are left to create
pub struct PreparedBatch {
    // streamed models carry their archetype so the streamer can be told when they are dropped
    pub archetype: Option<u32>,
    pub is_terrain: bool,
    pub is_box: bool,
    pub verts: Vec<GpuVertex>,
    pub insts: Vec<GpuInstance>,
    pub geoms: Vec<PreparedGeom>,
    pub cells: Vec<PreparedCell>,
    pub own_texture: Option<Texture2D>,
    pub tri_count: usize,
    pub geom_bytes: usize,
}

// placements of one model can span the whole map, sorting them by cell lets each cell be culled on its own
const BATCH_CELL: f32 = 1024.0;

// smooth normals, vertex and index data, cell sorted instances and their bounds
pub fn prepare_batch(b: WorldBatchInput, archetype: Option<u32>) -> Option<PreparedBatch> {
    let WorldBatchInput { mut mesh, instances: mut placed } = b;
    if mesh.verts.is_empty() || mesh.tris.is_empty() || placed.is_empty() {
        return None;
    }
    let use_u16 = mesh.verts.len() <= u16::MAX as usize;
    let geom_bytes = mesh.verts.len() * std::mem::size_of::<GpuVertex>()
        + mesh.tris.len() * if use_u16 { 6 } else { 12 }
        + placed.len() * std::mem::size_of::<GpuInstance>();

    let normals = compute_smooth_normals(&mesh);
    let cell_of = |p: Vec3| ((p.x / BATCH_CELL).floor() as i32, (p.y / BATCH_CELL).floor() as i32, (p.z / BATCH_CELL).floor() as i32);
    placed.sort_by_key(|i| cell_of(i.0));
    let insts: Vec<GpuInstance> = placed
        .iter()
        .map(|&(pos, rot, scale, col)| GpuInstance::from_mat4(Mat4::from_scale_rotation_translation(scale, rot, pos), col))
        .collect();

    // splits mesh3d by tri_geom material so every ttpf texture and paint group renders cleanly
    let own_texture = mesh.texture.take();
    let mut by_geom: std::collections::BTreeMap<u16, Vec<u32>> = std::collections::BTreeMap::new();
    for (ti, &[i0, i1, i2]) in mesh.tris.iter().enumerate() {
        let g = if own_texture.is_some() { 0 } else { mesh.tri_geom.get(ti).copied().unwrap_or(0) };
        let v = by_geom.entry(g).or_default();
        v.push(i0);
        v.push(i1);
        v.push(i2);
    }
    let geoms: Vec<PreparedGeom> = by_geom
        .into_iter()
        .map(|(g, indices)| PreparedGeom {
            indices: if use_u16 {
                PreparedIndices::U16(indices.iter().map(|&i| i as u16).collect())
            } else {
                PreparedIndices::U32(indices)
            },
            texture: if own_texture.is_some() { None } else { mesh.geom_materials.get(g as usize).and_then(|m| m.texture.clone()) },
        })
        .collect();

    let verts: Vec<GpuVertex> = mesh
        .verts
        .iter()
        .enumerate()
        .map(|(i, &pos)| GpuVertex {
            pos,
            normal: normals[i],
            uv: mesh.uvs.get(i).copied().unwrap_or([0.0, 0.0]),
        })
        .collect();

    let mut lmin = Vec3::splat(f32::MAX);
    let mut lmax = Vec3::splat(f32::MIN);
    for v in &mesh.verts {
        let p = Vec3::from_array(*v);
        lmin = lmin.min(p);
        lmax = lmax.max(p);
    }
    let local_center = (lmin + lmax) * 0.5;
    let local_rad = ((lmax - lmin) * 0.5).length();
    // one entry per run of placements in the same cell: first instance, count, bounds, largest copy
    let mut cells: Vec<PreparedCell> = Vec::new();
    let mut start = 0usize;
    while start < placed.len() {
        let key = cell_of(placed[start].0);
        let mut end = start;
        let mut wmin = Vec3::splat(f32::MAX);
        let mut wmax = Vec3::splat(f32::MIN);
        let mut obj_radius = 0.0f32;
        while end < placed.len() && cell_of(placed[end].0) == key {
            let (pos, rot, scale, _) = placed[end];
            let c = pos + rot * (local_center * scale);
            let r = local_rad * scale.abs().max_element();
            wmin = wmin.min(c - Vec3::splat(r));
            wmax = wmax.max(c + Vec3::splat(r));
            obj_radius = obj_radius.max(r);
            end += 1;
        }
        cells.push(PreparedCell {
            inst_first: start as u32,
            inst_count: (end - start) as u32,
            bound_center: (wmin + wmax) * 0.5,
            bound_radius: ((wmax - wmin) * 0.5).length().max(1.0),
            obj_radius: obj_radius.max(1.0),
        });
        start = end;
    }

    Some(PreparedBatch {
        archetype,
        is_terrain: mesh.name == "mhiu_world_terrain",
        is_box: mesh.name == "__placeholder_boxes__",
        verts,
        insts,
        geoms,
        cells,
        own_texture,
        tri_count: mesh.tris.len(),
        geom_bytes,
    })
}

const MAX_GEOM_BYTES: usize = 1024 * 1024 * 1024;
const MAX_TEX_BYTES: usize = 512 * 1024 * 1024;
// archetype the viewer let go of and the camera position at that moment
// an evicted one carries the distance it may load again within, a model turned away for lack of room carries none and is offered again right away
type UnloadNotice = (u32, Vec3, Option<f32>);
// an evicted model loads again once the camera is this much closer than where it was dropped
const RELOAD_STEP: f32 = 150.0;
// one dropped beyond its draw range also waits until it is this near to that range
const RELOAD_MARGIN: f32 = 200.0;
// eviction starts above the high mark and stops under the low one, both in percent of the geometry budget
const EVICT_HIGH_PCT: usize = 85;
const EVICT_LOW_PCT: usize = 65;
// a model has to stay undrawn this many frames before it can go
const EVICT_IDLE_FRAMES: u64 = 600;
// a freshly loaded model only gets this many frames to show up
const EVICT_FRESH_FRAMES: u64 = 30;

struct EvictCandidate {
    owner: u32,
    bytes: usize,
    last_drawn: u64,
    // distance from the camera to the nearest placement bounds
    dist: f32,
    // terrain, placeholder boxes and anything the streamer could not load again
    pinned: bool,
}

// owners to drop once geometry passes the high mark: idle ones only, farthest first, until under the low mark
fn select_evictions(cands: &[EvictCandidate], total_bytes: usize, frame: u64, high: usize, low: usize, idle_frames: u64) -> Vec<u32> {
    if total_bytes <= high {
        return Vec::new();
    }
    let mut idle: Vec<&EvictCandidate> = cands
        .iter()
        .filter(|c| !c.pinned && frame.saturating_sub(c.last_drawn) > idle_frames)
        .collect();
    idle.sort_by(|a, b| b.dist.total_cmp(&a.dist));
    let mut left = total_bytes;
    let mut out = Vec::new();
    for c in idle {
        if left <= low {
            break;
        }
        left = left.saturating_sub(c.bytes);
        out.push(c.owner);
    }
    out
}

pub enum WorldStreamMsg {
    Batch(PreparedBatch),
    MapCount(usize),
    TotalArchetypes(usize),
    ArchetypeDone,
}

pub fn launch_world_gpu_process(
    common_rpf: Option<&Path>,
    levels_rpf: Option<&Path>,
    ctx: eframe::egui::Context,
) -> Result<Receiver<Result<(), String>>, String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--gpu-world");
    if let Some(c) = common_rpf {
        cmd.arg("--common").arg(c);
    }
    if let Some(l) = levels_rpf {
        cmd.arg("--levels").arg(l);
    }
    use std::process::Stdio;
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let mut child = cmd.spawn().map_err(|e| format!("Failed to launch 3D GPU World: {e}"))?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        use std::io::{BufRead, BufReader, Read};
        let stderr = child.stderr.take().unwrap();
        let errors = std::thread::spawn(move || {
            let mut text = String::new();
            let _ = BufReader::new(stderr).read_to_string(&mut text);
            text
        });
        let mut ready = false;
        for line in BufReader::new(child.stdout.take().unwrap()).lines() {
            match line {
                Ok(line) if line == "WORLD_READY" => {
                    ready = true;
                    let _ = tx.send(Ok(()));
                    ctx.request_repaint();
                }
                Ok(line) if line.starts_with("WORLD_ERROR ") => {
                    let _ = tx.send(Err(line[12..].to_string()));
                    ctx.request_repaint();
                }
                Err(error) => {
                    let _ = tx.send(Err(format!("GPU viewer output failed: {error}")));
                    ctx.request_repaint();
                }
                _ => {}
            }
        }
        let status = child.wait();
        let errors = errors.join().unwrap_or_default();
        if !ready || status.as_ref().map_or(true, |s| !s.success()) {
            let _ = tx.send(Err(format!("GPU viewer exited ({status:?}): {}", errors.trim())));
            ctx.request_repaint();
        }
    });
    Ok(rx)
}

pub fn run_gpu_world_cli(common_rpf: Option<&Path>, levels_rpf: Option<&Path>) -> Result<(), String> {
    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn SetPriorityClass(hProcess: isize, dwPriorityClass: u32) -> i32;
        }
        // below_normal keeps windows audio and desktop apps stutter-free during streaming
        SetPriorityClass(GetCurrentProcess(), 0x0000_4000);
    }
    let fallback_dir = crate::load_last_folder();
    let resolved_common = common_rpf.map(Path::to_path_buf).or_else(|| {
        fallback_dir.as_ref().map(|d| d.join("common.rpf")).filter(|p| p.exists())
    });
    let resolved_levels = levels_rpf.map(Path::to_path_buf).or_else(|| {
        fallback_dir.as_ref().map(|d| d.join("levels.rpf")).filter(|p| p.exists())
    });
    if resolved_common.is_none() && resolved_levels.is_none() {
        return Err("No world archives found; open a game folder first".into());
    }
    let terrain = resolved_common.as_deref()
        .map(|p| crate::worldmap::load_mhiu_from_rpf(p).map(Arc::new)).transpose()?;
    let cfg = WorldLaunchConfig {
        terrain,
        batches: Vec::new(),
        levels_rpf: resolved_levels,
        map_count: 0,
        start_pos: [-127.276, -1855.924, 24.0],
    };
    let event_loop = EventLoop::builder().build().map_err(|e| format!("Cannot start GPU event loop: {e}"))?;
    let (tx, rx) = mpsc::sync_channel(16);
    let cam_track = Arc::new(std::sync::Mutex::new((Vec3::from_array(cfg.start_pos), 1u64)));
    let cam_track_bg = Arc::clone(&cam_track);
    // archetypes the viewer dropped or had no room for, see unload notice
    let (unload_tx, unload_rx) = mpsc::channel::<UnloadNotice>();
    if let Some(lp) = cfg.levels_rpf.clone() {
        std::thread::spawn(move || {
            // 1. fast-load brickell slod city blocks immediately
            if let Ok(mesh) = crate::model3d::decode_rage9_model_from_rpf(&lp, 8_679_358_480, "brickell_slod") {
                let slod = WorldBatchInput {
                    mesh,
                    instances: vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, [1.0; 4])],
                };
                if let Some(p) = prepare_batch(slod, None) {
                    let _ = tx.send(WorldStreamMsg::Batch(p));
                }
            }
            // 2. build/load cached 0DRPz map placements + model index
            eprintln!("[world-gpu] loading/building world index from {}", lp.display());
            let ix = crate::mapdata::build_or_load_world_index(&lp);
            eprintln!("[world-gpu] index ready: {} maps, {} models, {} entities", ix.map_count, ix.models.len(), ix.entities.len());
            let _ = tx.send(WorldStreamMsg::MapCount(ix.map_count));

            let mut grouped: HashMap<u32, (Vec3, Vec<(Vec3, Quat, Vec3, [f32; 4])>)> = HashMap::new();
            let mut box_insts: Vec<(Vec3, Quat, Vec3, [f32; 4])> = Vec::new();
            let start = cam_track_bg.lock().unwrap().0;
            for e in &ix.entities {
                let p = Vec3::from_array(e.pos);
                if !p.is_finite() || !e.scale_xy.is_finite() || !e.scale_z.is_finite() {
                    continue;
                }
                let q_raw = Quat::from_xyzw(e.rot[0], e.rot[1], e.rot[2], e.rot[3]);
                // e.rot is already conjugated to world space in mapdata.rs
                let q = if !q_raw.is_finite() || q_raw.length_squared() < 1e-6 {
                    Quat::IDENTITY
                } else {
                    q_raw.normalize()
                };
                if !ix.models.contains_key(&e.archetype_hash) {
                    if p.distance_squared(start) <= 2200.0 * 2200.0 && box_insts.len() < 12_000 {
                        let h = e.archetype_hash;
                        let r = 0.45 + ((h & 0xFF) as f32 / 255.0) * 0.45;
                        let g = 0.45 + (((h >> 8) & 0xFF) as f32 / 255.0) * 0.45;
                        let b = 0.50 + (((h >> 16) & 0xFF) as f32 / 255.0) * 0.45;
                        box_insts.push((p, q, Vec3::new(14.0 * e.scale_xy, 14.0 * e.scale_xy, 18.0 * e.scale_z), [r, g, b, 1.0]));
                    }
                    continue;
                }
                let sc = Vec3::new(e.scale_xy, e.scale_xy, e.scale_z);
                let entry = grouped.entry(e.archetype_hash).or_insert_with(|| (p, Vec::new()));
                entry.1.push((p, q, sc, [1.0; 4]));
            }
            if !box_insts.is_empty() {
                let bmesh = Mesh3D::from_parts(
                    "__placeholder_boxes__".into(),
                    vec![
                        [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0],
                        [-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [-0.5, 0.5, 1.0],
                    ],
                    vec![[0.0, 0.0]; 8],
                    vec![
                        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
                    ],
                    vec![0; 12],
                    Vec::new(),
                    None,
                    String::new(),
                    1,
                    0,
                    "BOX".into(),
                );
                if let Some(p) = prepare_batch(WorldBatchInput { mesh: bmesh, instances: box_insts }, None) {
                    let _ = tx.send(WorldStreamMsg::Batch(p));
                }
            }

            let all_groups: Vec<(u32, u64, Vec<(Vec3, Quat, Vec3, [f32; 4])>)> = grouped
                .into_iter()
                .filter_map(|(h, (_rep, insts))| ix.models.get(&h).map(|&moff| (h, moff, insts)))
                .collect();
            // 200m spatial grid for O(cells) distance-tiered streaming up to 6 km
            const STREAM_CELL: f32 = 200.0;
            let mut stream_grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
            for (idx, (_, _, insts)) in all_groups.iter().enumerate() {
                for &(p, _, _, _) in insts {
                    let key = ((p.x / STREAM_CELL).floor() as i32, (p.y / STREAM_CELL).floor() as i32);
                    let cell_vec = stream_grid.entry(key).or_default();
                    if cell_vec.last().copied() != Some(idx) {
                        cell_vec.push(idx);
                    }
                }
            }
            for v in stream_grid.values_mut() {
                v.sort_unstable();
                v.dedup();
            }
            eprintln!("[world-gpu] {} unique drawable archetypes across {} 200m cells indexed for live streaming", all_groups.len(), stream_grid.len());
            let group_of: HashMap<u32, usize> = all_groups.iter().enumerate().map(|(idx, g)| (g.0, idx)).collect();
            // squared distance from a point to the 200m cell that holds p
            let cell_d2 = |p: Vec3, from: Vec3| {
                let x0 = (p.x / STREAM_CELL).floor() * STREAM_CELL;
                let y0 = (p.y / STREAM_CELL).floor() * STREAM_CELL;
                let rx = (x0 - from.x).max(0.0).max(from.x - (x0 + STREAM_CELL));
                let ry = (y0 - from.y).max(0.0).max(from.y - (y0 + STREAM_CELL));
                rx * rx + ry * ry
            };

            let mut loaded = std::collections::HashSet::<u32>::new();
            // how near the camera has to be before a dropped archetype is decoded again
            let mut reload_within: HashMap<u32, f32> = HashMap::new();
            let mut done_count = 0usize;
            let mut last_gen = 0u64;
            loop {
                let (cam_p, cur_gen) = *cam_track_bg.lock().unwrap();
                if cur_gen == last_gen {
                    std::thread::sleep(std::time::Duration::from_millis(80));
                    continue;
                }
                last_gen = cur_gen;
                // dropped archetypes may load again once the camera comes back
                while let Ok((h, at, hold)) = unload_rx.try_recv() {
                    loaded.remove(&h);
                    reload_within.remove(&h);
                    if let (Some(limit), Some(&idx)) = (hold, group_of.get(&h)) {
                        // loading it back at the distance it was dropped at would only get it dropped again
                        let d2 = all_groups[idx].2.iter().map(|i| cell_d2(i.0, at)).fold(f32::MAX, f32::min);
                        reload_within.insert(h, (d2.sqrt() - RELOAD_STEP).max(0.0).min(limit));
                    }
                }
                let max_r = 6000.0f32;
                let cr = (max_r / STREAM_CELL).ceil() as i32;
                let cx0 = (cam_p.x / STREAM_CELL).floor() as i32;
                let cy0 = (cam_p.y / STREAM_CELL).floor() as i32;
                let mut cells: Vec<((i32, i32), f32)> = Vec::new();
                for dx in -cr..=cr {
                    for dy in -cr..=cr {
                        let key = (cx0 + dx, cy0 + dy);
                        if !stream_grid.contains_key(&key) {
                            continue;
                        }
                        let x0 = key.0 as f32 * STREAM_CELL;
                        let y0 = key.1 as f32 * STREAM_CELL;
                        let rx = (x0 - cam_p.x).max(0.0).max(cam_p.x - (x0 + STREAM_CELL));
                        let ry = (y0 - cam_p.y).max(0.0).max(cam_p.y - (y0 + STREAM_CELL));
                        let d2 = rx * rx + ry * ry;
                        if d2 <= max_r * max_r {
                            cells.push((key, d2));
                        }
                    }
                }
                cells.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                let take_n = if loaded.is_empty() { 1200 } else { 500 };
                let mut seen_idx = std::collections::HashSet::new();
                let mut batch_list: Vec<(u32, u64, Vec<(Vec3, Quat, Vec3, [f32; 4])>)> = Vec::with_capacity(take_n);
                'outer: for (key, d2) in cells {
                    if let Some(indices) = stream_grid.get(&key) {
                        for &idx in indices {
                            let (h, moff, ref insts) = all_groups[idx];
                            if loaded.contains(&h) || !seen_idx.insert(idx) {
                                continue;
                            }
                            // cells come nearest first so this is the closest the model gets in this pass
                            if reload_within.get(&h).is_some_and(|&r| d2.sqrt() > r) {
                                continue;
                            }
                            batch_list.push((h, moff, insts.clone()));
                            if batch_list.len() >= take_n {
                                break 'outer;
                            }
                        }
                    }
                }
                if batch_list.is_empty() {
                    continue;
                }
                // counted by finished decodes, the loaded set shrinks when the viewer drops models
                let _ = tx.send(WorldStreamMsg::TotalArchetypes(done_count + batch_list.len()));
                let groups = Arc::new(batch_list);
                let cursor = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                let workers = std::thread::available_parallelism().map(|n| (n.get() / 2).clamp(3, 6)).unwrap_or(3);
                // a camera move stops the batch early and whatever was not reached is queued again
                let mut finished_count = 0usize;
                let worker_done: Vec<Vec<u32>> = std::thread::scope(|s| {
                    let handles: Vec<_> = (0..workers)
                        .map(|_| {
                            let groups = Arc::clone(&groups);
                            let cursor = Arc::clone(&cursor);
                            let tx = tx.clone();
                            let lp = lp.clone();
                            let cam_track_bg = Arc::clone(&cam_track_bg);
                            s.spawn(move || {
                                // the viewer shows these at 128 pixels so larger mips are never read
                                crate::tex::set_thread_texture_cap(128);
                                let mut local_done = Vec::new();
                                loop {
                                    if cam_track_bg.lock().unwrap().1 != last_gen {
                                        break;
                                    }
                                    let i = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    if i >= groups.len() {
                                        break;
                                    }
                                    let (h, moff, ref insts) = groups[i];
                                    if let Ok(mesh) = crate::model3d::decode_rage9_model_from_rpf(&lp, moff, &format!("arch_{h:08x}")) {
                                        let lo = mesh.name.to_ascii_lowercase();
                                        if !(lo.ends_with("_shadow") || lo.ends_with("_pxy") || lo.ends_with("_lit_proxy")) {
                                            // normals, buffers and cell bounds are built here so the render thread only uploads
                                            if let Some(p) = prepare_batch(WorldBatchInput { mesh, instances: insts.clone() }, Some(h)) {
                                                if tx.send(WorldStreamMsg::Batch(p)).is_err() {
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    local_done.push(h);
                                    if tx.send(WorldStreamMsg::ArchetypeDone).is_err() {
                                        break;
                                    }
                                }
                                local_done
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
                });
                for chunk in worker_done {
                    finished_count += chunk.len();
                    loaded.extend(chunk);
                }
                done_count += finished_count;
                if finished_count < groups.len() {
                    // let the next pass run even if the camera has stopped moving again
                    last_gen = 0;
                }
                let _ = tx.send(WorldStreamMsg::TotalArchetypes(done_count));
            }
        });
    }
    let mut app = GpuWorldApp::new(cfg, rx, cam_track, unload_tx);
    event_loop.run_app(&mut app).map_err(|e| format!("GPU event loop failed: {e}"))?;
    match app.failure { Some(error) => Err(error), None => Ok(()) }
}

const WORLD_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    flags: vec4<f32>,
    map_y: vec4<f32>,
    hud: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@group(1) @binding(0) var albedo_tex: texture_2d<f32>;
@group(1) @binding(1) var albedo_samp: sampler;

@group(2) @binding(0) var shadow_tex: texture_depth_2d;
@group(2) @binding(1) var shadow_samp: sampler_comparison;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) m0: vec4<f32>,
    @location(4) m1: vec4<f32>,
    @location(5) m2: vec4<f32>,
    @location(6) m3: vec4<f32>,
    @location(7) inst_col: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) inst_col: vec4<f32>,
    @location(4) shadow_coord: vec4<f32>,
};

@vertex
fn vs_shadow(in: VsIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let wp = model * vec4<f32>(in.pos, 1.0);
    return u.light_vp * wp;
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let wp = (model * vec4<f32>(in.pos, 1.0)).xyz;
    let wn = normalize((model * vec4<f32>(in.normal, 0.0)).xyz);
    var out: VsOut;
    out.clip_pos = u.view_proj * vec4<f32>(wp, 1.0);
    out.world_pos = wp;
    out.world_n = wn;
    out.uv = in.uv;
    out.inst_col = in.inst_col;
    out.shadow_coord = u.light_vp * vec4<f32>(wp, 1.0);
    return out;
}

fn sample_shadow(sc: vec4<f32>, ndl: f32) -> f32 {
    let proj = sc.xyz / max(sc.w, 0.0001);
    let uv = proj.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0) {
        return 1.0;
    }
    let bias = max(0.0015 * (1.0 - ndl), 0.0003);
    let texel = 1.0 / 2048.0;
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            sum += textureSampleCompare(
                shadow_tex,
                shadow_samp,
                uv + vec2<f32>(f32(x), f32(y)) * texel,
                proj.z - bias
            );
        }
    }
    return sum / 9.0;
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn neutral_col(c: vec3<f32>, k: f32) -> vec3<f32> {
    return mix(c, vec3<f32>(dot(c, vec3<f32>(0.2126, 0.7152, 0.0722))), k);
}

// cook-torrance pbr lighting adapted from sixng shader.wgsl
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex_c = textureSample(albedo_tex, albedo_samp, in.uv);
    if (tex_c.a < 0.4) {
        discard;
    }
    let albedo = tex_c.rgb * in.inst_col.rgb;
    let n = normalize(in.world_n);
    let l = normalize(u.sun_dir.xyz);
    let v = normalize(u.cam_pos.xyz - in.world_pos);
    let h = normalize(l + v);

    let ndl = max(dot(n, l), 0.0);
    let ndh = max(dot(n, h), 0.0);
    let ndv = max(dot(n, v), 0.001);

    var sh = sample_shadow(in.shadow_coord, ndl);
    let rt_on = u.flags.x > 0.5;
    if (rt_on) {
        sh = smoothstep(0.08, 0.92, sh);
    }

    let rough = 0.62;
    let metal = 0.06;
    let a2 = rough * rough * rough * rough;
    let dd = ndh * ndh * (a2 - 1.0) + 1.0;
    let d_ggx = a2 / (3.14159 * dd * dd);
    let k = (rough + 1.0) * (rough + 1.0) / 8.0;
    let g_smith = (ndl / (ndl * (1.0 - k) + k)) * (ndv / (ndv * (1.0 - k) + k));
    let f0 = mix(vec3<f32>(0.04), albedo, metal);
    let fr = f0 + (1.0 - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    let spec = d_ggx * g_smith * fr / max(4.0 * ndl * ndv, 0.001);
    let kd = (1.0 - fr) * (1.0 - metal);

    let sun_elev = clamp(l.z * 2.2, 0.0, 1.0);
    let sun_col = mix(vec3<f32>(1.0, 0.48, 0.22), vec3<f32>(1.05, 0.98, 0.90), sun_elev) * (2.2 * sun_elev + 0.18);
    let direct = (kd * albedo / 3.14159 + spec) * sun_col * ndl * sh * 2.4;

    let hemi = n.z * 0.5 + 0.5;
    let amb_d = mix(neutral_col(u.sky_horizon.rgb, 0.78) * 0.55, neutral_col(u.sky_zenith.rgb, 0.78), hemi);
    let fr_v = f0 + (max(vec3<f32>(1.0 - rough), f0) - f0) * pow(1.0 - ndv, 5.0);
    let env = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, 0.5) * 0.45;
    let rt_bounce = select(0.0, 0.14 * (1.0 - ndv) * (0.4 + 0.6 * hemi), rt_on);

    let lit = direct + kd * albedo * amb_d + env * fr_v + amb_d * rt_bounce;

    let dist = length(u.cam_pos.xyz - in.world_pos);
    let fog = 1.0 - exp(-dist * 0.000028 * max(u.hud.w, 0.0));
    let fog_col = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, 0.25);
    let col = mix(lit, fog_col, clamp(fog, 0.0, 0.65)) * max(u.sky_zenith.w, 0.2);
    return vec4<f32>(aces_tonemap(col), 1.0);
}

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    return vec4<f32>(pts[vid], 0.9999, 1.0);
}

@fragment
fn fs_sky(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let t = clamp(1.0 - frag_pos.y / max(u.map_y.w, 1.0), 0.0, 1.0);
    let col = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, pow(t, 0.7));
    return vec4<f32>(aces_tonemap(col), 1.0);
}

struct MinimapOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_minimap(@builtin(vertex_index) vid: u32) -> MinimapOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    );
    let q = quad[vid];
    let w = max(u.map_y.z, 640.0);
    let h = max(u.map_y.w, 480.0);
    let px = 18.0 + q.x * 190.0;
    let py = h - 18.0 - 190.0 + q.y * 190.0;
    let ndc_x = (px / w) * 2.0 - 1.0;
    let ndc_y = 1.0 - (py / h) * 2.0;
    var out: MinimapOut;
    out.pos = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = vec2<f32>(q.x, 1.0 - q.y);
    return out;
}

@fragment
fn fs_minimap(in: MinimapOut) -> @location(0) vec4<f32> {
    let c = textureSample(albedo_tex, albedo_samp, in.uv);
    let border = in.uv.x < 0.02 || in.uv.x > 0.98 || in.uv.y < 0.02 || in.uv.y > 0.98;
    if (border) {
        return vec4<f32>(0.92, 0.94, 0.98, 0.95);
    }
    let player_u = (u.cam_pos.x - u.flags.z) / max(u.flags.w, 1.0);
    let player_v = 1.0 - (u.cam_pos.y - u.map_y.x) / max(u.map_y.y, 1.0);
    let d = length(in.uv - vec2<f32>(player_u, player_v));
    if (d < 0.028) {
        return vec4<f32>(1.0, 0.66, 0.11, 1.0);
    }
    return vec4<f32>(c.rgb, 0.92);
}

struct WaterOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_water(@builtin(vertex_index) vid: u32) -> WaterOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
    );
    let p = quad[vid] * 26000.0;
    let wp = vec3<f32>(p.x, p.y, -0.5);
    var out: WaterOut;
    out.clip_pos = u.view_proj * vec4<f32>(wp, 1.0);
    out.world_pos = wp;
    return out;
}

@fragment
fn fs_water(in: WaterOut) -> @location(0) vec4<f32> {
    let dist = length(u.cam_pos.xyz - in.world_pos);
    let wave_fade = exp(-dist * 0.0015);
    let t = u.hud.z;
    let w1 = sin(in.world_pos.x * 0.11 + t * 1.4) * cos(in.world_pos.y * 0.13 - t * 1.1);
    let w2 = sin((in.world_pos.x * 0.07 - in.world_pos.y * 0.09) - t * 0.8);
    let n = normalize(vec3<f32>((w1 + w2) * 0.035 * wave_fade, (w1 - w2) * 0.035 * wave_fade, 1.0));
    let v = normalize(u.cam_pos.xyz - in.world_pos);
    let l = normalize(u.sun_dir.xyz);
    let h = normalize(l + v);
    let fres = pow(1.0 - max(dot(n, v), 0.0), 4.0);
    let sun_elev = clamp(l.z * 2.2, 0.0, 1.0);
    let deep = vec3<f32>(0.02, 0.11, 0.19) * (0.35 + 0.65 * sun_elev);
    let sky_r = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, 0.45);
    let glint = pow(max(dot(n, h), 0.0), 140.0) * 1.6 * sun_elev;
    let water_col = mix(deep, sky_r, clamp(fres * 0.85 + 0.08, 0.0, 0.92)) + vec3<f32>(glint);
    let fog = clamp(1.0 - exp(-dist * 0.00010), 0.0, 0.90);
    let fog_col = mix(u.sky_horizon.rgb, u.sky_zenith.rgb, 0.25);
    return vec4<f32>(aces_tonemap(mix(water_col, fog_col, fog)), 1.0);
}

struct ProgOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_progress(@builtin(vertex_index) vid: u32) -> ProgOut {
    var quad = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
    );
    let q = quad[vid];
    let w = max(u.map_y.z, 640.0);
    let h = max(u.map_y.w, 480.0);
    let bw = min(460.0, w * 0.5);
    let bh = 14.0;
    let px = (w - bw) * 0.5 + q.x * bw;
    let py = 18.0 + q.y * bh;
    var out: ProgOut;
    out.pos = vec4<f32>((px / w) * 2.0 - 1.0, 1.0 - (py / h) * 2.0, 0.0, 1.0);
    out.uv = q;
    return out;
}

@fragment
fn fs_progress(in: ProgOut) -> @location(0) vec4<f32> {
    if (u.hud.y < 0.5) {
        discard;
    }
    let border = in.uv.x < 0.006 || in.uv.x > 0.994 || in.uv.y < 0.12 || in.uv.y > 0.88;
    if (border) {
        return vec4<f32>(0.85, 0.90, 0.96, 0.85);
    }
    if (in.uv.x <= clamp(u.hud.x, 0.0, 1.0)) {
        return vec4<f32>(0.22, 0.80, 0.98, 0.95);
    }
    return vec4<f32>(0.08, 0.10, 0.14, 0.80);
}
"#;

struct GpuMeshBatch {
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    inst_buf: wgpu::Buffer,
    index_count: u32,
    index_format: wgpu::IndexFormat,
    inst_first: u32,
    inst_count: u32,
    tex_bind_group: wgpu::BindGroup,
    bound_center: Vec3,
    bound_radius: f32,
    // radius of one placed copy, decides draw distance and shadow casting
    obj_radius: f32,
    is_box: bool,
    is_terrain: bool,
    // every batch made from one model shares this id
    owner: u32,
}

// bookkeeping of one loaded model so it can be dropped as a whole
struct LoadedModel {
    // only streamed archetypes can be loaded again, everything else stays for good
    archetype: Option<u32>,
    geom_bytes: usize,
    tris: usize,
    instances: usize,
    last_drawn: u64,
    // bytes of the texture only this model uses
    own_tex_bytes: usize,
    // one upload cache key per material that uses a shared texture
    tex_keys: Vec<usize>,
}

// one cached upload with the decoded texture kept alive behind it
struct SharedTex {
    bg: wgpu::BindGroup,
    // holding the texture keeps its address from being reused by another one
    _keep: Arc<Texture2D>,
    bytes: usize,
    // materials of loaded models that draw with it
    users: u32,
}

struct GpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    uniform_buf: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    shadow_bg: wgpu::BindGroup,
    tex_bgl: wgpu::BindGroupLayout,
    tex_samp: wgpu::Sampler,
    white_bg: wgpu::BindGroup,
    shadow_pipeline: wgpu::RenderPipeline,
    main_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    minimap_pipeline: wgpu::RenderPipeline,
    progress_pipeline: wgpu::RenderPipeline,
    minimap_bg: wgpu::BindGroup,
    egui_renderer: egui_wgpu::Renderer,
    batches: Vec<GpuMeshBatch>,
    models: HashMap<u32, LoadedModel>,
    // owner id of every loaded streamed archetype
    by_archetype: HashMap<u32, u32>,
    next_owner: u32,
    // archetypes dropped or turned away since the streamer was last told, with the reload limit of the unload notice
    unloaded: Vec<(u32, Option<f32>)>,
    next_evict_check: u64,
    // smoke scenes shrink these to reach eviction with a handful of models
    geom_budget: usize,
    evict_idle_frames: u64,
    tri_count: usize,
    instance_count: usize,
    model_count: usize,
    map_count: usize,
    stream_done: usize,
    stream_total: usize,
    tex_bytes: usize,
    geom_bytes: usize,
    vram_oom: bool,
    // texture bind groups by pixel buffer address
    tex_uploads: HashMap<usize, SharedTex>,
    frame_idx: u64,
    start_instant: Instant,
    has_hw_rt: bool,
    map_min: [f32; 2],
    map_span: [f32; 2],
}

fn create_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> wgpu::Texture {
    
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world_depth"),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bgl: &wgpu::BindGroupLayout,
    samp: &wgpu::Sampler,
    tex: &Texture2D,
    max_dim: usize,
) -> (wgpu::BindGroup, usize) {
    let w0 = tex.w.max(1);
    let h0 = tex.h.max(1);
    let scale = (w0.max(h0) + max_dim - 1) / max_dim.max(1);
    let (w, h, raw) = if scale > 1 && tex.pixels.len() >= w0 * h0 {
        let nw = (w0 / scale).max(1);
        let nh = (h0 / scale).max(1);
        let mut buf = Vec::with_capacity(nw * nh * 4);
        for y in 0..nh {
            let sy = (y * scale).min(h0 - 1);
            for x in 0..nw {
                let sx = (x * scale).min(w0 - 1);
                buf.extend_from_slice(&tex.pixels[sy * w0 + sx].to_array());
            }
        }
        (nw, nh, buf)
    } else {
        let mut buf = Vec::with_capacity(tex.pixels.len() * 4);
        for c in &tex.pixels {
            buf.extend_from_slice(&c.to_array());
        }
        (w0, h0, buf)
    };
    let bytes = raw.len();
    let gpu_tex = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("albedo"),
            size: wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &raw,
    );
    let view = gpu_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tex_bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(samp) },
        ],
    });
    (bg, bytes)
}

// how far from the camera a placed copy of this size is still drawn
fn draw_dist(obj_radius: f32, draw_radius: f32) -> f32 {
    let base_d: f32 = if obj_radius > 35.0 { 6500.0 } else if obj_radius < 2.0 { 450.0 } else { 3200.0 };
    base_d.min(draw_radius).max(if obj_radius > 35.0 { draw_radius } else { 450.0 })
}

fn compute_smooth_normals(mesh: &Mesh3D) -> Vec<[f32; 3]> {
    let mut accum = vec![Vec3::ZERO; mesh.verts.len()];
    for (ti, &[a, b, c]) in mesh.tris.iter().enumerate() {
        let n = mesh.normals.get(ti).copied().unwrap_or([0.0, 0.0, 1.0]);
        let nv = Vec3::from_array(n);
        if (a as usize) < accum.len() { accum[a as usize] += nv; }
        if (b as usize) < accum.len() { accum[b as usize] += nv; }
        if (c as usize) < accum.len() { accum[c as usize] += nv; }
    }
    accum
        .into_iter()
        .map(|n| {
            let nn = n.normalize_or_zero();
            if nn.length_squared() > 0.01 { nn.to_array() } else { [0.0, 0.0, 1.0] }
        })
        .collect()
}

impl GpuState {
    fn new(window: Arc<Window>, cfg: WorldLaunchConfig) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No GPU adapter found");

        let has_hw_rt = adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        let mut req_features = wgpu::Features::empty();
        if has_hw_rt {
            req_features |= wgpu::Features::EXPERIMENTAL_RAY_QUERY;
        }
        let (device, queue) = pollster_block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("percystar_world_device"),
                required_features: req_features,
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_texture = create_depth_texture(&device, config.width, config.height);
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sun_shadow_map"),
            size: wgpu::Extent3d { width: 2048, height: 2048, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let shadow_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let u_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("u_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("u_bg"),
            layout: &u_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() }],
        });

        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let shadow_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow_bg"),
            layout: &shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shadow_samp) },
            ],
        });

        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tex_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let tex_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world_shader"),
            source: wgpu::ShaderSource::Wgsl(WORLD_WGSL.into()),
        });

        let v_layouts = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GpuVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GpuInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4],
            },
        ];

        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pl"),
            bind_group_layouts: &[&u_bgl, &tex_bgl],
            push_constant_ranges: &[],
        });

        let main_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main_pl"),
            bind_group_layouts: &[&u_bgl, &tex_bgl, &shadow_bgl],
            push_constant_ranges: &[],
        });

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipe"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_shadow"), buffers: &v_layouts, compilation_options: Default::default() },
            fragment: None,
            primitive: wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let main_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("main_pipe"),
            layout: Some(&main_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &v_layouts, compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky_pipe"),
            layout: Some(&main_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_sky"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water_pipe"),
            layout: Some(&main_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_water"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_water"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let minimap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("minimap_pipe"),
            layout: Some(&main_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_minimap"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_minimap"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let progress_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("progress_pipe"),
            layout: Some(&main_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_progress"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_progress"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let white_tex = Texture2D::new(1, 1, vec![eframe::egui::Color32::from_rgb(190, 195, 205)]);
        let (white_bg, _) = upload_texture(&device, &queue, &tex_bgl, &tex_samp, &white_tex, 256);
        let (minimap_bg, _) = upload_texture(&device, &queue, &tex_bgl, &tex_samp, &white_tex, 256);
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

        let mut st = Self {
            window,
            surface,
            device,
            queue,
            config,
            depth_view,
            depth_texture,
            shadow_view,
            uniform_buf,
            uniform_bg,
            shadow_bg,
            tex_bgl,
            tex_samp,
            white_bg,
            shadow_pipeline,
            main_pipeline,
            sky_pipeline,
            water_pipeline,
            minimap_pipeline,
            progress_pipeline,
            minimap_bg,
            egui_renderer,
            batches: Vec::new(),
            models: HashMap::new(),
            by_archetype: HashMap::new(),
            next_owner: 0,
            unloaded: Vec::new(),
            next_evict_check: 0,
            geom_budget: MAX_GEOM_BYTES,
            evict_idle_frames: EVICT_IDLE_FRAMES,
            tri_count: 0,
            instance_count: 0,
            model_count: 0,
            map_count: cfg.map_count,
            stream_done: 0,
            stream_total: 0,
            tex_bytes: 0,
            geom_bytes: 0,
            vram_oom: false,
            tex_uploads: HashMap::new(),
            frame_idx: 0,
            start_instant: Instant::now(),
            has_hw_rt,
            map_min: [-13000.0, -9000.0],
            map_span: [20000.0, 20000.0],
        };

        if let Some(ref terrain) = cfg.terrain {
            st.map_min = [terrain.bmin[0], terrain.bmin[1]];
            st.map_span = [terrain.bmax[0] - terrain.bmin[0], terrain.bmax[1] - terrain.bmin[1]];
            let tmesh = terrain.build_world_terrain_mesh(8);
            if let Some(ref t) = tmesh.texture {
                let (mbg, bytes) = upload_texture(&st.device, &st.queue, &st.tex_bgl, &st.tex_samp, t, 1024);
                st.minimap_bg = mbg;
                st.tex_bytes += bytes;
            }
            st.add_batch(WorldBatchInput {
                mesh: tmesh,
                instances: vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, [1.0; 4])],
            });
            st.model_count = st.model_count.saturating_sub(1);
        }

        for b in cfg.batches {
            st.add_batch(b);
        }
        st
    }

    // prepares on the calling thread, for the terrain and the batches handed over at launch
    fn add_batch(&mut self, b: WorldBatchInput) {
        if let Some(p) = prepare_batch(b, None) {
            self.add_prepared(p);
        }
    }

    // gpu side of a prepared model: buffers, texture bind groups and one draw batch per material and cell
    fn add_prepared(&mut self, p: PreparedBatch) {
        if p.archetype.is_some_and(|h| self.by_archetype.contains_key(&h)) {
            return;
        }
        if !p.is_terrain && (self.vram_oom || self.geom_bytes.saturating_add(p.geom_bytes) > self.geom_budget) {
            // no room right now, the streamer may offer it again once the camera is closer
            if !self.vram_oom {
                self.unloaded.extend(p.archetype.map(|h| (h, None)));
            }
            return;
        }

        self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let inst_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("m_inst"),
            contents: bytemuck::cast_slice(&p.insts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("m_vb"),
            contents: bytemuck::cast_slice(&p.verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibs: Vec<wgpu::Buffer> = p
            .geoms
            .iter()
            .map(|g| {
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("m_ib"),
                    contents: g.indices.bytes(),
                    usage: wgpu::BufferUsages::INDEX,
                })
            })
            .collect();
        if let Some(err) = pollster_block_on(self.device.pop_error_scope()) {
            self.vram_oom = true;
            eprintln!(
                "[world-gpu] GPU VRAM limit reached ({} MiB geom, {} MiB tex): {err}",
                self.geom_bytes / (1024 * 1024),
                self.tex_bytes / (1024 * 1024)
            );
            return;
        }

        let owner = self.next_owner;
        self.next_owner = self.next_owner.wrapping_add(1);
        let mut model = LoadedModel {
            archetype: p.archetype,
            geom_bytes: p.geom_bytes,
            tris: p.tri_count * p.insts.len(),
            instances: p.insts.len(),
            // a fresh model starts most of the idle window back, one that never shows is the first to go
            last_drawn: self.frame_idx.saturating_sub(self.evict_idle_frames.saturating_sub(EVICT_FRESH_FRAMES)),
            own_tex_bytes: 0,
            tex_keys: Vec::new(),
        };

        for (geom, ib) in p.geoms.iter().zip(ibs) {
            let tex_bg = if let Some(t) = p.own_texture.as_ref().filter(|_| self.tex_bytes < MAX_TEX_BYTES) {
                let (bg, bytes) = upload_texture(&self.device, &self.queue, &self.tex_bgl, &self.tex_samp, t, 128);
                self.tex_bytes += bytes;
                model.own_tex_bytes += bytes;
                bg
            } else if let Some(t) = geom.texture.as_ref() {
                // materials that share a decoded texture share one upload
                let key = t.pixels.as_ptr() as usize;
                if !self.tex_uploads.contains_key(&key) && self.tex_bytes < MAX_TEX_BYTES {
                    let (bg, bytes) = upload_texture(&self.device, &self.queue, &self.tex_bgl, &self.tex_samp, t, 128);
                    self.tex_bytes += bytes;
                    self.tex_uploads.insert(key, SharedTex { bg, _keep: Arc::clone(t), bytes, users: 0 });
                }
                match self.tex_uploads.get_mut(&key) {
                    Some(shared) => {
                        shared.users += 1;
                        model.tex_keys.push(key);
                        shared.bg.clone()
                    }
                    None => self.white_bg.clone(),
                }
            } else {
                self.white_bg.clone()
            };

            for c in &p.cells {
                self.batches.push(GpuMeshBatch {
                    vb: vb.clone(),
                    ib: ib.clone(),
                    inst_buf: inst_buf.clone(),
                    index_count: geom.indices.len() as u32,
                    index_format: geom.indices.format(),
                    inst_first: c.inst_first,
                    inst_count: c.inst_count,
                    tex_bind_group: tex_bg.clone(),
                    bound_center: c.bound_center,
                    bound_radius: c.bound_radius,
                    obj_radius: c.obj_radius,
                    is_box: p.is_box,
                    is_terrain: p.is_terrain,
                    owner,
                });
            }
        }

        self.geom_bytes += model.geom_bytes;
        self.tri_count += model.tris;
        self.instance_count += model.instances;
        self.model_count += 1;
        if let Some(h) = model.archetype {
            self.by_archetype.insert(h, owner);
        }
        self.models.insert(owner, model);
    }

    // drops idle far away models once geometry passes the high mark, their archetypes go to the unloaded list
    fn evict_idle(&mut self, cam: Vec3, draw_radius: f32) {
        let high = self.geom_budget / 100 * EVICT_HIGH_PCT;
        let low = self.geom_budget / 100 * EVICT_LOW_PCT;
        if self.geom_bytes <= high || self.frame_idx < self.next_evict_check {
            return;
        }
        // nothing may be idle yet, look again a little later instead of every frame
        self.next_evict_check = self.frame_idx + 20;
        // per owner: distance to the nearest placement bounds, widest draw range, whether any cell is inside its range
        let mut reach: HashMap<u32, (f32, f32, bool)> = HashMap::new();
        for b in &self.batches {
            let d = ((b.bound_center - cam).length() - b.bound_radius).max(0.0);
            let range = draw_dist(b.obj_radius, draw_radius);
            let e = reach.entry(b.owner).or_insert((f32::MAX, 0.0, false));
            e.0 = e.0.min(d);
            e.1 = e.1.max(range);
            e.2 |= d <= range || b.bound_radius >= 5000.0;
        }
        let cands: Vec<EvictCandidate> = self
            .models
            .iter()
            .map(|(&owner, m)| EvictCandidate {
                owner,
                bytes: m.geom_bytes,
                last_drawn: m.last_drawn,
                dist: reach.get(&owner).map_or(f32::MAX, |r| r.0),
                pinned: m.archetype.is_none(),
            })
            .collect();
        let victims: std::collections::HashSet<u32> =
            select_evictions(&cands, self.geom_bytes, self.frame_idx, high, low, self.evict_idle_frames).into_iter().collect();
        if victims.is_empty() {
            return;
        }
        // the buffers and unshared bind groups are released with the last batch that holds them
        self.batches.retain(|b| !victims.contains(&b.owner));
        for owner in &victims {
            let Some(m) = self.models.remove(owner) else { continue };
            self.geom_bytes = self.geom_bytes.saturating_sub(m.geom_bytes);
            self.tri_count = self.tri_count.saturating_sub(m.tris);
            self.instance_count = self.instance_count.saturating_sub(m.instances);
            self.model_count = self.model_count.saturating_sub(1);
            self.tex_bytes = self.tex_bytes.saturating_sub(m.own_tex_bytes);
            for key in m.tex_keys {
                // a shared upload goes only when no loaded model draws with it any more
                let Some(shared) = self.tex_uploads.get_mut(&key) else { continue };
                shared.users = shared.users.saturating_sub(1);
                if shared.users == 0 {
                    self.tex_bytes = self.tex_bytes.saturating_sub(shared.bytes);
                    self.tex_uploads.remove(&key);
                }
            }
            if let Some(h) = m.archetype {
                self.by_archetype.remove(&h);
                // out of view inside its draw range it only has to come closer, beyond the range it also has to get near it
                let limit = match reach.get(owner) {
                    Some(&(_, range, false)) => range + RELOAD_MARGIN,
                    _ => f32::MAX,
                };
                self.unloaded.push((h, Some(limit)));
            }
        }
    }

    // stamps the owner of a drawn batch, batches of one model sit next to each other so repeats are skipped
    fn mark_drawn(models: &mut HashMap<u32, LoadedModel>, last_owner: &mut Option<u32>, owner: u32, frame: u64) {
        if *last_owner != Some(owner) {
            *last_owner = Some(owner);
            if let Some(m) = models.get_mut(&owner) {
                m.last_drawn = frame;
            }
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth_texture = create_depth_texture(&self.device, w, h);
        self.depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
    }
}

struct GpuWorldApp {
    cfg: Option<WorldLaunchConfig>,
    terrain_ref: Option<Arc<WorldMapData>>,
    stream_rx: Receiver<WorldStreamMsg>,
    cam_track: Arc<std::sync::Mutex<(Vec3, u64)>>,
    // tells the streamer which archetypes are gone
    unload_tx: mpsc::Sender<UnloadNotice>,
    last_stream_cam: Vec3,
    gpu: Option<GpuState>,
    egui_ctx: eframe::egui::Context,
    egui_events: Vec<eframe::egui::Event>,
    pointer_pos: eframe::egui::Pos2,
    cam_pos: Vec3,
    vel_z: f32,
    yaw: f32,
    pitch: f32,
    time_h: f32,
    auto_sun: bool,
    rt_on: bool,
    walk_mode: bool,
    menu_open: bool,
    show_models: bool,
    show_boxes: bool,
    show_water: bool,
    draw_radius: f32,
    fog_density: f32,
    exposure: f32,
    keys: [bool; 9], // W A S D Q E Shift Space Ctrl
    dragging: bool,
    last_cursor: Option<PhysicalPosition<f64>>,
    last_frame: Instant,
    last_title: Instant,
    smoke_frames: Option<u32>,
    // geometry budget and idle window for the eviction smoke scene
    smoke_budget: Option<(usize, u64)>,
    ready: bool,
    failure: Option<String>,
}

impl GpuWorldApp {
    fn new(
        cfg: WorldLaunchConfig,
        stream_rx: Receiver<WorldStreamMsg>,
        cam_track: Arc<std::sync::Mutex<(Vec3, u64)>>,
        unload_tx: mpsc::Sender<UnloadNotice>,
    ) -> Self {
        Self {
            cam_pos: Vec3::from_array(cfg.start_pos),
            last_stream_cam: Vec3::from_array(cfg.start_pos),
            terrain_ref: cfg.terrain.clone(),
            cfg: Some(cfg),
            stream_rx,
            cam_track,
            unload_tx,
            gpu: None,
            egui_ctx: eframe::egui::Context::default(),
            egui_events: Vec::new(),
            pointer_pos: eframe::egui::Pos2::ZERO,
            vel_z: 0.0,
            yaw: 0.65,
            pitch: -0.08,
            time_h: 13.50,
            auto_sun: false,
            rt_on: true,
            walk_mode: false,
            menu_open: true,
            show_models: true,
            show_boxes: false,
            show_water: true,
            draw_radius: 6000.0,
            fog_density: 0.70,
            exposure: 1.0,
            keys: [false; 9],
            dragging: false,
            last_cursor: None,
            last_frame: Instant::now(),
            last_title: Instant::now(),
            smoke_frames: None,
            smoke_budget: None,
            ready: false,
            failure: None,
        }
    }

    fn sample_terrain_z(&self, x: f32, y: f32) -> f32 {
        let Some(ref t) = self.terrain_ref else { return 0.0 };
        let u = ((x - t.bmin[0]) / (t.bmax[0] - t.bmin[0]).max(1.0)).clamp(0.0, 1.0);
        let v = (1.0 - (y - t.bmin[1]) / (t.bmax[1] - t.bmin[1]).max(1.0)).clamp(0.0, 1.0);
        let ix = ((u * (t.w.saturating_sub(1)) as f32) as usize).min(t.w - 1);
        let iy = ((v * (t.h.saturating_sub(1)) as f32) as usize).min(t.h - 1);
        let h = t.heights[iy * t.w + ix] as f32 / 255.0;
        t.bmin[2] + h * (t.bmax[2] - t.bmin[2])
    }
}

impl ApplicationHandler for GpuWorldApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let icon = image::load_from_memory(include_bytes!("../assets/logo.png")).ok().and_then(|image| {
                let image = image.into_rgba8();
                let (w, h) = image.dimensions();
                Icon::from_rgba(image.into_raw(), w, h).ok()
            });
        let attrs = Window::default_attributes()
            .with_title("PercyStar 3D World (wgpu)")
            .with_window_icon(icon.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 860.0));
        #[cfg(windows)]
        let attrs = {
            use winit::platform::windows::WindowAttributesExtWindows;
            attrs.with_taskbar_icon(icon)
        };
        let win = match event_loop.create_window(attrs) {
            Ok(win) => win,
            Err(error) => {
                self.failure = Some(format!("Could not create GPU World window: {error}"));
                event_loop.exit();
                return;
            }
        };
        let win = Arc::new(win);
        let cfg = self.cfg.take().unwrap_or(WorldLaunchConfig {
            terrain: None,
            batches: Vec::new(),
            levels_rpf: None,
            map_count: 0,
            start_pos: [0.0, 0.0, 120.0],
        });
        let mut gpu = GpuState::new(win, cfg);
        if let Some((budget, idle_frames)) = self.smoke_budget {
            gpu.geom_budget = budget;
            gpu.evict_idle_frames = idle_frames;
        }
        gpu.window.request_redraw();
        self.gpu = Some(gpu);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(sz) => {
                if let Some(g) = &mut self.gpu {
                    g.resize(sz.width, sz.height);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                let egui_btn = match button {
                    MouseButton::Left => Some(eframe::egui::PointerButton::Primary),
                    MouseButton::Right => Some(eframe::egui::PointerButton::Secondary),
                    MouseButton::Middle => Some(eframe::egui::PointerButton::Middle),
                    _ => None,
                };
                if let Some(eb) = egui_btn {
                    self.egui_events.push(eframe::egui::Event::PointerButton {
                        pos: self.pointer_pos,
                        button: eb,
                        pressed,
                        modifiers: Default::default(),
                    });
                }
                let over_ui = self.menu_open && self.egui_ctx.wants_pointer_input();
                if matches!(button, MouseButton::Left | MouseButton::Right) && (!pressed || !over_ui) {
                    self.dragging = pressed;
                    if !self.dragging {
                        self.last_cursor = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let scale = self.gpu.as_ref().map(|g| g.window.scale_factor() as f32).unwrap_or(1.0);
                self.pointer_pos = eframe::egui::pos2(position.x as f32 / scale, position.y as f32 / scale);
                self.egui_events.push(eframe::egui::Event::PointerMoved(self.pointer_pos));
                if self.dragging {
                    if let Some(prev) = self.last_cursor {
                        let dx = (position.x - prev.x) as f32;
                        let dy = (position.y - prev.y) as f32;
                        self.yaw -= dx * 0.004;
                        self.pitch = (self.pitch - dy * 0.004).clamp(-1.45, 1.45);
                    }
                    self.last_cursor = Some(position);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let (dx, dy) = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => (x * 24.0, y * 24.0),
                    winit::event::MouseScrollDelta::PixelDelta(p) => (p.x as f32, p.y as f32),
                };
                self.egui_events.push(eframe::egui::Event::MouseWheel {
                    unit: eframe::egui::MouseWheelUnit::Point,
                    delta: eframe::egui::vec2(dx, dy),
                    modifiers: Default::default(),
                });
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let down = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => self.keys[0] = down,
                        KeyCode::KeyA => self.keys[1] = down,
                        KeyCode::KeyS => self.keys[2] = down,
                        KeyCode::KeyD => self.keys[3] = down,
                        KeyCode::KeyQ | KeyCode::KeyC => self.keys[4] = down,
                        KeyCode::KeyE => self.keys[5] = down,
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys[6] = down,
                        KeyCode::Space => self.keys[7] = down,
                        KeyCode::ControlLeft | KeyCode::ControlRight => self.keys[8] = down,
                        KeyCode::Home | KeyCode::F1 if down => self.menu_open = !self.menu_open,
                        KeyCode::KeyB if down => self.show_boxes = !self.show_boxes,
                        KeyCode::KeyV if down => self.show_water = !self.show_water,
                        KeyCode::F6 if down => self.auto_sun = !self.auto_sun,
                        KeyCode::KeyR if down => self.rt_on = !self.rt_on,
                        KeyCode::KeyG if down => {
                            self.walk_mode = !self.walk_mode;
                            self.vel_z = 0.0;
                        }
                        KeyCode::BracketRight if down => self.time_h = (self.time_h + 0.5) % 24.0,
                        KeyCode::BracketLeft if down => self.time_h = (self.time_h + 23.5) % 24.0,
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().min(0.1);
                self.last_frame = now;

                if let Some(g) = &mut self.gpu {
                    // room is made first so the uploads below are not turned away
                    // a lowered radius setting must not cost models the default radius would show
                    g.evict_idle(self.cam_pos, self.draw_radius.max(6000.0));
                    let upload_start = Instant::now();
                    for _ in 0..8 {
                        if upload_start.elapsed().as_micros() >= 6000 {
                            break;
                        }
                        let Ok(msg) = self.stream_rx.try_recv() else { break };
                        match msg {
                            WorldStreamMsg::Batch(batch) => g.add_prepared(batch),
                            WorldStreamMsg::MapCount(mc) => g.map_count = mc,
                            WorldStreamMsg::TotalArchetypes(t) => g.stream_total = t,
                            WorldStreamMsg::ArchetypeDone => g.stream_done += 1,
                        }
                    }
                    for (h, hold) in g.unloaded.drain(..) {
                        let _ = self.unload_tx.send((h, self.cam_pos, hold));
                    }
                }

                if self.auto_sun {
                    self.time_h = (self.time_h + dt * 0.35) % 24.0;
                }
                let fwd = Vec3::new(self.yaw.cos() * self.pitch.cos(), self.yaw.sin() * self.pitch.cos(), self.pitch.sin()).normalize();
                let fwd_flat = Vec3::new(self.yaw.cos(), self.yaw.sin(), 0.0).normalize();
                let right = fwd.cross(Vec3::Z).normalize_or(Vec3::Y);
                let move_fwd = if self.walk_mode { fwd_flat } else { fwd };
                let speed = if self.walk_mode {
                    if self.keys[6] { 45.0 } else if self.keys[8] { 4.0 } else { 14.0 }
                } else if self.keys[6] {
                    650.0
                } else if self.keys[8] {
                    28.0
                } else {
                    140.0
                } * dt;
                if self.keys[0] { self.cam_pos += move_fwd * speed; }
                if self.keys[2] { self.cam_pos -= move_fwd * speed; }
                if self.keys[3] { self.cam_pos += right * speed; }
                if self.keys[1] { self.cam_pos -= right * speed; }
                if self.walk_mode {
                    let ground_z = self.sample_terrain_z(self.cam_pos.x, self.cam_pos.y) + 1.8;
                    self.vel_z -= 18.0 * dt;
                    self.cam_pos.z += self.vel_z * dt;
                    if self.cam_pos.z <= ground_z {
                        self.cam_pos.z = ground_z;
                        self.vel_z = if self.keys[7] { 8.5 } else { 0.0 };
                    }
                } else {
                    if self.keys[5] { self.cam_pos.z += speed; }
                    if self.keys[4] { self.cam_pos.z -= speed; }
                }

                if self.cam_pos.distance_squared(self.last_stream_cam) > 180.0 * 180.0 {
                    self.last_stream_cam = self.cam_pos;
                    if let Ok(mut ct) = self.cam_track.lock() {
                        ct.0 = self.cam_pos;
                        ct.1 += 1;
                    }
                }

                let Some(g) = &mut self.gpu else { return };
                let az = (self.time_h / 24.0) * std::f32::consts::TAU - 1.2;
                let el = ((self.time_h - 6.0) / 12.0 * std::f32::consts::PI).sin() * 1.15;
                let sun_dir = Vec3::new(az.cos() * el.cos(), az.sin() * el.cos(), el.sin()).normalize();

                if now - self.last_title >= std::time::Duration::from_millis(120) {
                    self.last_title = now;
                    let rt_lbl = if !self.rt_on { "OFF" } else if g.has_hw_rt { "HW-RT" } else { "SS-RT/GI" };
                    let mode_lbl = if self.walk_mode { "WALK(G)" } else { "FLY(G)" };
                    let load_lbl = if g.stream_total > 0 && g.stream_done < g.stream_total {
                        format!(" | Load {}/{} ({:.0}%)", g.stream_done, g.stream_total, (g.stream_done as f32 / g.stream_total as f32) * 100.0)
                    } else if g.stream_total > 0 {
                        " | World Ready".to_string()
                    } else {
                        " | Indexing...".to_string()
                    };
                    g.window.set_title(&format!(
                        "PercyStar 3D World | pos {:.0} {:.0} {:.0} | {} | Sun az {:.1} el {:+.1} (F6) | Time {:.2}h | {} maps | {} models | {} instances | {:.2} M tris | RT: {}{}",
                        self.cam_pos.x, self.cam_pos.y, self.cam_pos.z, mode_lbl,
                        az.to_degrees().rem_euclid(360.0), el.to_degrees(),
                        self.time_h, g.map_count, g.model_count, g.instance_count,
                        g.tri_count as f64 / 1e6, rt_lbl, load_lbl
                    ));
                }

                let aspect = g.config.width as f32 / g.config.height.max(1) as f32;
                let proj = Mat4::perspective_rh(60f32.to_radians(), aspect, 0.5, 45000.0);
                let view = Mat4::look_to_rh(self.cam_pos, fwd, Vec3::Z);
                let view_proj = proj * view;

                let light_target = self.cam_pos + fwd * 300.0;
                let light_eye = light_target + sun_dir * 3500.0;
                let light_view = Mat4::look_at_rh(light_eye, light_target, Vec3::Z);
                let light_proj = Mat4::orthographic_rh(-1800.0, 1800.0, -1800.0, 1800.0, 50.0, 9000.0);
                let light_vp = light_proj * light_view;

                let day_fac = (sun_dir.z * 2.5).clamp(0.0, 1.0);
                let sky_zenith = [0.06 + day_fac * 0.20, 0.09 + day_fac * 0.42, 0.16 + day_fac * 0.72, self.exposure];
                let sky_horizon = [0.14 + day_fac * 0.54, 0.16 + day_fac * 0.62, 0.24 + day_fac * 0.68, 1.0];
                let load_prog = if g.stream_total > 0 { (g.stream_done as f32 / g.stream_total as f32).clamp(0.0, 1.0) } else { 0.02 };
                let load_active = if g.stream_total == 0 || g.stream_done < g.stream_total { 1.0 } else { 0.0 };

                let uni = FrameUniforms {
                    view_proj: view_proj.to_cols_array_2d(),
                    light_vp: light_vp.to_cols_array_2d(),
                    cam_pos: [self.cam_pos.x, self.cam_pos.y, self.cam_pos.z, 1.0],
                    sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
                    sky_zenith,
                    sky_horizon,
                    flags: [if self.rt_on { 1.0 } else { 0.0 }, self.time_h, g.map_min[0], g.map_span[0]],
                    map_y: [g.map_min[1], g.map_span[1], g.config.width as f32, g.config.height as f32],
                    hud: [load_prog, load_active, g.start_instant.elapsed().as_secs_f32(), self.fog_density],
                };
                g.queue.write_buffer(&g.uniform_buf, 0, bytemuck::bytes_of(&uni));

                let Ok(frame) = g.surface.get_current_texture() else { return };
                let view_tex = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut enc = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("world_enc") });
                g.frame_idx = g.frame_idx.wrapping_add(1);

                if g.frame_idx % 2 == 1 {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("shadow_pass"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &g.shadow_view,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rp.set_pipeline(&g.shadow_pipeline);
                    rp.set_bind_group(0, &g.uniform_bg, &[]);
                    // casting a shadow counts as drawn, the caster may sit just outside the view
                    let mut last_owner = None;
                    for b in &g.batches {
                        if b.obj_radius < 3.0 {
                            continue;
                        }
                        if b.bound_radius < 5000.0 && (b.bound_center - light_target).length_squared() > (1200.0 + b.bound_radius).powi(2) {
                            continue;
                        }
                        GpuState::mark_drawn(&mut g.models, &mut last_owner, b.owner, g.frame_idx);
                        rp.set_bind_group(1, &b.tex_bind_group, &[]);
                        rp.set_vertex_buffer(0, b.vb.slice(..));
                        rp.set_vertex_buffer(1, b.inst_buf.slice(..));
                        rp.set_index_buffer(b.ib.slice(..), b.index_format);
                        rp.draw_indexed(0..b.index_count, 0, b.inst_first..b.inst_first + b.inst_count);
                    }
                }

                {
                    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("main_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view_tex,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &g.depth_view,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    rp.set_pipeline(&g.sky_pipeline);
                    rp.set_bind_group(0, &g.uniform_bg, &[]);
                    rp.set_bind_group(1, &g.minimap_bg, &[]);
                    rp.set_bind_group(2, &g.shadow_bg, &[]);
                    rp.draw(0..3, 0..1);

                    // Smoke checks must not mistake the ocean plane for loaded geometry.
                    if self.smoke_frames.is_none() && self.show_water {
                        rp.set_pipeline(&g.water_pipeline);
                        rp.draw(0..6, 0..1);
                    }

                    rp.set_pipeline(&g.main_pipeline);
                    let mut drawn = 0usize;
                    let mut last_owner = None;
                    for b in &g.batches {
                        if b.is_box && !self.show_boxes {
                            continue;
                        }
                        if !b.is_box && !b.is_terrain && !self.show_models {
                            continue;
                        }
                        if !b.is_terrain && b.bound_radius < 5000.0 {
                            let to_b = b.bound_center - self.cam_pos;
                            if to_b.dot(fwd) < -(b.bound_radius + 120.0) {
                                continue;
                            }
                            let d2 = to_b.length_squared();
                            let max_d = draw_dist(b.obj_radius, self.draw_radius);
                            if d2 > (max_d + b.bound_radius).powi(2) {
                                continue;
                            }
                            drawn += 1;
                            if drawn > 6000 {
                                break;
                            }
                        }
                        GpuState::mark_drawn(&mut g.models, &mut last_owner, b.owner, g.frame_idx);
                        rp.set_bind_group(1, &b.tex_bind_group, &[]);
                        rp.set_vertex_buffer(0, b.vb.slice(..));
                        rp.set_vertex_buffer(1, b.inst_buf.slice(..));
                        rp.set_index_buffer(b.ib.slice(..), b.index_format);
                        rp.draw_indexed(0..b.index_count, 0, b.inst_first..b.inst_first + b.inst_count);
                    }

                    rp.set_pipeline(&g.minimap_pipeline);
                    rp.set_bind_group(1, &g.minimap_bg, &[]);
                    rp.draw(0..6, 0..1);

                    rp.set_pipeline(&g.progress_pipeline);
                    rp.draw(0..6, 0..1);
                }

                if self.menu_open {
                    let scale = g.window.scale_factor() as f32;
                    let screen_desc = egui_wgpu::ScreenDescriptor {
                        size_in_pixels: [g.config.width, g.config.height],
                        pixels_per_point: scale,
                    };
                    let raw_in = eframe::egui::RawInput {
                        screen_rect: Some(eframe::egui::Rect::from_min_size(
                            eframe::egui::Pos2::ZERO,
                            eframe::egui::vec2(g.config.width as f32 / scale, g.config.height as f32 / scale),
                        )),
                        events: std::mem::take(&mut self.egui_events),
                        ..Default::default()
                    };
                    let full_out = self.egui_ctx.run(raw_in, |ctx| {
                        let mut vis = eframe::egui::Visuals::dark();
                        vis.window_fill = eframe::egui::Color32::from_rgba_premultiplied(20, 13, 31, 238);
                        vis.panel_fill = eframe::egui::Color32::from_rgb(20, 13, 31);
                        vis.window_stroke = eframe::egui::Stroke::new(1.2_f32, eframe::egui::Color32::from_rgb(255, 72, 165));
                        vis.widgets.inactive.bg_fill = eframe::egui::Color32::from_rgb(36, 22, 54);
                        vis.widgets.hovered.bg_fill = eframe::egui::Color32::from_rgb(56, 32, 82);
                        vis.widgets.active.bg_fill = eframe::egui::Color32::from_rgb(255, 168, 28);
                        vis.selection.bg_fill = eframe::egui::Color32::from_rgb(255, 168, 28);
                        ctx.set_visuals(vis);

                        eframe::egui::Window::new(
                            eframe::egui::RichText::new("PERCYSTAR · WORLD & RESHADE (Home / F1)")
                                .strong()
                                .color(eframe::egui::Color32::from_rgb(255, 168, 28)),
                        )
                        .default_pos([g.config.width as f32 / scale - 370.0, 18.0])
                        .default_width(350.0)
                        .resizable(false)
                        .show(ctx, |ui| {
                            ui.label(
                                eframe::egui::RichText::new(format!(
                                    "{} maps · {}/{} models · {} inst · {:.2}M tris · {} MB VRAM",
                                    g.map_count, g.model_count, g.stream_total, g.instance_count,
                                    g.tri_count as f64 / 1e6, g.tex_bytes / (1024 * 1024)
                                ))
                                .size(11.0)
                                .color(eframe::egui::Color32::from_rgb(95, 215, 255)),
                            );
                            ui.separator();
                            ui.add(eframe::egui::Slider::new(&mut self.time_h, 0.0..=24.0).text("Time of Day (h)"));
                            ui.horizontal(|ui| {
                                for (lbl, h) in [("Dawn", 6.2), ("Noon", 12.5), ("Golden", 18.0), ("Night", 23.0)] {
                                    if ui.small_button(lbl).clicked() {
                                        self.time_h = h;
                                        self.auto_sun = false;
                                    }
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Radius:");
                                for (lbl, r) in [("700m", 700.0), ("1.5km", 1500.0), ("3km", 3000.0), ("6km", 6000.0), ("16km", 16000.0)] {
                                    if ui.selectable_label((self.draw_radius - r).abs() < 1.0, lbl).clicked() {
                                        self.draw_radius = r;
                                    }
                                }
                            });
                            ui.add(eframe::egui::Slider::new(&mut self.fog_density, 0.0..=2.0).text("Fog Density"));
                            ui.add(eframe::egui::Slider::new(&mut self.exposure, 0.3..=2.2).text("Exposure"));
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.auto_sun, "Auto Sun (F6)");
                                ui.checkbox(&mut self.rt_on, "RT Shadows (R)");
                                ui.checkbox(&mut self.show_water, "Water (V)");
                            });
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.show_models, "Real Models");
                                ui.checkbox(&mut self.show_boxes, "Boxes (B)");
                                ui.checkbox(&mut self.walk_mode, "Walk (G)");
                            });
                            ui.horizontal(|ui| {
                                let btn = eframe::egui::Button::new(
                                    eframe::egui::RichText::new("Go to Downtown Brickell")
                                        .strong()
                                        .color(eframe::egui::Color32::BLACK),
                                )
                                .fill(eframe::egui::Color32::from_rgb(255, 168, 28));
                                if ui.add(btn).clicked() {
                                    self.cam_pos = Vec3::new(-127.276, -1855.924, 24.0);
                                    self.yaw = 0.65;
                                    self.pitch = -0.08;
                                }
                                ui.label(
                                    eframe::egui::RichText::new(format!("Pos: {:.0}, {:.0}, {:.0}", self.cam_pos.x, self.cam_pos.y, self.cam_pos.z))
                                        .size(11.0)
                                        .color(eframe::egui::Color32::from_rgb(200, 190, 220)),
                                );
                            });
                        });
                    });
                    for (id, delta) in &full_out.textures_delta.set {
                        g.egui_renderer.update_texture(&g.device, &g.queue, *id, delta);
                    }
                    let tris = self.egui_ctx.tessellate(full_out.shapes, scale);
                    g.egui_renderer.update_buffers(&g.device, &g.queue, &mut enc, &tris, &screen_desc);
                    {
                        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("egui_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view_tex,
                                resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        }).forget_lifetime();
                        g.egui_renderer.render(&mut rp, &tris, &screen_desc);
                    }
                    for id in &full_out.textures_delta.free {
                        g.egui_renderer.free_texture(id);
                    }
                } else {
                    self.egui_events.clear();
                }

                g.queue.submit(Some(enc.finish()));
                frame.present();
                if !self.ready {
                    use std::io::Write;
                    println!("WORLD_READY");
                    let _ = std::io::stdout().flush();
                    self.ready = true;
                }
                if let Some(frames) = &mut self.smoke_frames {
                    if !g.batches.is_empty() {
                        *frames -= 1;
                        if *frames == 0 {
                            g.assert_visible_geometry();
                            event_loop.exit();
                        }
                    }
                }
                g.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn pollster_block_on<F: std::future::Future>(fut: F) -> F::Output {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn dummy_raw() -> RawWaker {
        fn no_op(_: *const ()) {}
        fn clone(p: *const ()) -> RawWaker { RawWaker::new(p, &VTABLE) }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    let waker = unsafe { Waker::from_raw(dummy_raw()) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        if let Poll::Ready(res) = fut.as_mut().poll(&mut cx) {
            return res;
        }
        std::thread::yield_now();
    }
}

#[cfg(test)]
mod terrain_sampling_tests {
    use super::*;
    #[test]
    fn camera_keeps_configured_start_and_samples_north_to_south() {
        let cfg = WorldLaunchConfig {
            terrain: Some(Arc::new(WorldMapData {
                w: 2,
                h: 2,
                bmin: [-10.0, -10.0, 0.0],
                bmax: [10.0, 10.0, 255.0],
                heights: vec![10, 10, 200, 200],
                tex: None,
            })),
            batches: vec![],
            levels_rpf: None,
            map_count: 1,
            start_pos: [-127.276, -1855.924, 24.0],
        };
        let (_, rx) = mpsc::sync_channel(16);
        let start = Vec3::from_array(cfg.start_pos);
        let app = GpuWorldApp::new(cfg, rx, Arc::new(std::sync::Mutex::new((start, 1))), mpsc::channel().0);
        assert_eq!(app.cam_pos, start);
        assert!((app.sample_terrain_z(0.0, 10.0) - 10.0).abs() < 0.001);
        assert!((app.sample_terrain_z(0.0, -10.0) - 200.0).abs() < 0.001);
    }
}

#[cfg(test)]
mod startup_tests {
    use super::*;
    #[test]
    fn missing_world_archive_reports_an_error() {
        let missing = Path::new("target/missing-world-regression.rpf");
        assert!(!missing.exists());
        assert!(run_gpu_world_cli(Some(missing), None).is_err());
    }
}

#[cfg(test)]
mod prepare_tests {
    use super::*;

    fn quad(tri_geom: Vec<u16>, texture: Option<Texture2D>) -> Mesh3D {
        Mesh3D::from_parts(
            "quad".into(),
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            vec![[0, 1, 2], [1, 3, 2]],
            tri_geom,
            vec![],
            texture,
            String::new(),
            2,
            1,
            String::new(),
        )
    }

    fn at(x: f32, scale: f32) -> (Vec3, Quat, Vec3, [f32; 4]) {
        (Vec3::new(x, 10.0, 0.0), Quat::IDENTITY, Vec3::splat(scale), [1.0; 4])
    }

    #[test]
    fn splits_materials_and_sorts_instances_into_cells() {
        // the far placement comes first so the cell sort has to move it behind the two near ones
        let input = WorldBatchInput { mesh: quad(vec![0, 1], None), instances: vec![at(2100.0, 2.0), at(10.0, 1.0), at(30.0, 1.0)] };
        let p = prepare_batch(input, Some(0xABCD)).unwrap();
        assert_eq!(p.archetype, Some(0xABCD));
        assert!(!p.is_terrain && !p.is_box && p.own_texture.is_none());
        assert_eq!((p.verts.len(), p.insts.len(), p.tri_count), (4, 3, 2));
        assert_eq!(p.geom_bytes, 4 * 32 + 2 * 6 + 3 * 80);
        assert_eq!(p.verts[3].pos, [1.0, 1.0, 0.0]);
        assert_eq!(p.verts[3].uv, [1.0, 1.0]);
        assert_eq!(p.verts[3].normal, [0.0, 0.0, 1.0]);

        assert_eq!(p.geoms.len(), 2);
        assert!(matches!(&p.geoms[0].indices, PreparedIndices::U16(v) if v == &[0, 1, 2]));
        assert!(matches!(&p.geoms[1].indices, PreparedIndices::U16(v) if v == &[1, 3, 2]));
        assert_eq!(p.geoms[0].indices.format(), wgpu::IndexFormat::Uint16);
        assert_eq!(p.geoms[0].indices.bytes().len(), 6);

        assert_eq!(p.insts.iter().map(|i| i.model_3[0]).collect::<Vec<_>>(), [10.0, 30.0, 2100.0]);
        assert_eq!(p.cells.len(), 2);
        let (near, far) = (&p.cells[0], &p.cells[1]);
        assert_eq!((near.inst_first, near.inst_count), (0, 2));
        assert_eq!((far.inst_first, far.inst_count), (2, 1));
        // the quad is centred on 0.5 0.5 with a half diagonal of sqrt 0.5
        let r = 0.5f32.sqrt();
        assert!(near.bound_center.distance(Vec3::new(20.5, 10.5, 0.0)) < 1e-3);
        assert!((near.bound_radius - Vec3::new(10.0 + r, r, r).length()).abs() < 1e-3);
        assert_eq!(near.obj_radius, 1.0);
        assert!(far.bound_center.distance(Vec3::new(2101.0, 11.0, 0.0)) < 1e-3);
        assert!((far.bound_radius - Vec3::splat(2.0 * r).length()).abs() < 1e-3);
        assert!((far.obj_radius - 2.0 * r).abs() < 1e-4);
    }

    #[test]
    fn large_meshes_use_wide_indices() {
        let n = u16::MAX as usize + 2;
        let mesh = Mesh3D::from_parts(
            "big".into(),
            (0..n).map(|i| [i as f32, (i % 7) as f32, 0.0]).collect(),
            Vec::new(),
            vec![[0, 1, n as u32 - 1]],
            vec![0],
            vec![],
            None,
            String::new(),
            1,
            1,
            String::new(),
        );
        let p = prepare_batch(WorldBatchInput { mesh, instances: vec![at(0.0, 1.0)] }, None).unwrap();
        assert!(matches!(&p.geoms[0].indices, PreparedIndices::U32(v) if v == &[0, 1, n as u32 - 1]));
        assert_eq!(p.geoms[0].indices.format(), wgpu::IndexFormat::Uint32);
        assert_eq!(p.geom_bytes, n * 32 + 12 + 80);
        assert_eq!(p.verts[5].uv, [0.0, 0.0]);
    }

    #[test]
    fn own_texture_keeps_one_material_and_empty_input_is_skipped() {
        let tex = Texture2D::new(1, 1, vec![eframe::egui::Color32::WHITE]);
        let p = prepare_batch(WorldBatchInput { mesh: quad(vec![0, 1], Some(tex)), instances: vec![at(0.0, 1.0)] }, None).unwrap();
        assert!(p.own_texture.is_some());
        assert_eq!(p.geoms.len(), 1);
        assert_eq!(p.geoms[0].indices.len(), 6);
        assert!(p.geoms[0].texture.is_none());
        assert!(prepare_batch(WorldBatchInput { mesh: quad(vec![0, 1], None), instances: Vec::new() }, None).is_none());
    }
}

#[cfg(test)]
mod eviction_tests {
    use super::*;

    fn cand(owner: u32, bytes: usize, last_drawn: u64, dist: f32) -> EvictCandidate {
        EvictCandidate { owner, bytes, last_drawn, dist, pinned: false }
    }

    #[test]
    fn idle_models_go_farthest_first_until_the_low_mark() {
        let mut cands = vec![
            cand(1, 10, 100, 500.0),
            cand(2, 10, 100, 4000.0),
            // drawn a moment ago
            cand(3, 10, 900, 9000.0),
            cand(4, 10, 100, 2000.0),
            cand(5, 10, 0, 20000.0),
            cand(6, 10, 100, 100.0),
        ];
        // terrain and placeholder boxes stay whatever their age and distance
        cands[4].pinned = true;
        assert_eq!(select_evictions(&cands, 90, 1000, 85, 65, 600), [2, 4, 1]);
        // nothing happens until the high mark is passed
        assert!(select_evictions(&cands, 85, 1000, 85, 65, 600).is_empty());
    }

    #[test]
    fn recently_drawn_models_are_never_chosen() {
        // exactly at the idle window still counts as recent
        let cands = [cand(1, 50, 400, 8000.0), cand(2, 50, 999, 9000.0)];
        assert!(select_evictions(&cands, 100, 1000, 85, 65, 600).is_empty());
        assert_eq!(select_evictions(&cands, 100, 1001, 85, 65, 600), [1]);
    }
}
