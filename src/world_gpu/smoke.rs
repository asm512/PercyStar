use super::*;

fn terrain() -> Arc<WorldMapData> {
    Arc::new(WorldMapData {
        w: 64,
        h: 64,
        bmin: [-13000.0, -9000.0, 0.0],
        bmax: [7000.0, 11000.0, 20.0],
        heights: vec![128; 4096],
        tex: None,
    })
}

fn model() -> WorldBatchInput {
    let mut mesh = Mesh3D::from_parts(
        "material smoke scene".into(),
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        vec![[0, 1, 2], [1, 3, 2]],
        vec![0, 1],
        vec![],
        None,
        String::new(),
        2,
        1,
        String::new(),
    );
    mesh.geom_materials = [[230, 60, 40], [40, 200, 80]]
        .into_iter()
        .map(|c| {
            let mut material = crate::model3d::GeomMaterial::default();
            material.texture = Some(Arc::new(Texture2D::new(1, 1, vec![eframe::egui::Color32::from_rgb(c[0], c[1], c[2])])));
            material
        })
        .collect();
    WorldBatchInput {
        mesh,
        instances: vec![(
            Vec3::new(-70.0, -1820.0, 5.0),
            Quat::IDENTITY,
            Vec3::splat(50.0),
            [1.0; 4],
        )],
    }
}

const EVICT_SMOKE_BUDGET: usize = 2200;

pub fn run_gpu_world_smoke(scene: &str) -> Result<(), String> {
    let mut cfg = WorldLaunchConfig {
        terrain: None,
        batches: Vec::new(),
        levels_rpf: None,
        map_count: 0,
        start_pos: [-127.276, -1855.924, 24.0],
    };
    let (tx, rx) = mpsc::sync_channel(16);
    let (unload_tx, unload_rx) = mpsc::channel::<UnloadNotice>();
    // archetypes the viewer reported back and whether they were evicted or only turned away
    let dropped = Arc::new(std::sync::Mutex::new(Vec::<(u32, bool)>::new()));
    let start_pos = Vec3::from_array(cfg.start_pos);
    match scene {
        "terrain" => {
            cfg.terrain = Some(terrain());
            cfg.map_count = 1;
        }
        "model" => cfg.batches.push(model()),
        "stream" => {
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if let Some(p) = prepare_batch(model(), None) {
                    let _ = tx.send(WorldStreamMsg::Batch(p));
                }
            });
        }
        "evict" => {
            // the launch batch stays for good, archetype 0 streams in next to it and the rest far behind the camera
            cfg.batches.push(model());
            let dropped = Arc::clone(&dropped);
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(100));
                for h in 0..12u32 {
                    let mut m = model();
                    if h > 0 {
                        m.instances[0].0 = Vec3::new(-9000.0 - 500.0 * h as f32, -8000.0, 5.0);
                    }
                    if let Some(p) = prepare_batch(m, Some(h)) {
                        let _ = tx.send(WorldStreamMsg::Batch(p));
                    }
                }
                let mut reloaded = false;
                while let Ok((h, _, hold)) = unload_rx.recv() {
                    dropped.lock().unwrap().push((h, hold.is_some()));
                    // the first evicted archetype comes back in view the way the streamer would bring it
                    if hold.is_some() && !reloaded {
                        reloaded = true;
                        if let Some(p) = prepare_batch(model(), Some(h)) {
                            let _ = tx.send(WorldStreamMsg::Batch(p));
                        }
                    }
                }
            });
        }
        _ => return Err(format!("Unknown smoke scene: {scene}")),
    }
    let event_loop = EventLoop::new().map_err(|e| e.to_string())?;
    let cam_track = Arc::new(std::sync::Mutex::new((start_pos, 1u64)));
    let mut app = GpuWorldApp::new(cfg, rx, cam_track, unload_tx);
    app.menu_open = false;
    app.smoke_frames = Some(3);
    if scene == "evict" {
        // room for ten of the 220 byte models and a two frame idle window
        app.smoke_budget = Some((EVICT_SMOKE_BUDGET, 2));
        app.smoke_frames = Some(90);
    }
    event_loop.run_app(&mut app).map_err(|e| e.to_string())?;
    if app.cam_pos != start_pos {
        return Err("Loading geometry changed the configured camera position".into());
    }
    if scene == "evict" {
        let gpu = app.gpu.as_ref().ok_or("GPU did not initialize")?;
        let dropped = dropped.lock().unwrap();
        let evicted: Vec<u32> = dropped.iter().filter(|d| d.1).map(|d| d.0).collect();
        if evicted.is_empty() || evicted.contains(&0) {
            return Err(format!("Eviction did not pick the idle far models: {evicted:?}"));
        }
        if !gpu.by_archetype.contains_key(&0) || !gpu.by_archetype.contains_key(&evicted[0]) {
            return Err("The drawn model or the reloaded one is missing".into());
        }
        if evicted[1..].iter().any(|h| gpu.by_archetype.contains_key(h)) {
            return Err("An evicted model is still loaded".into());
        }
        let n = gpu.models.len();
        let bytes: usize = gpu.models.values().map(|m| m.geom_bytes).sum();
        if gpu.geom_bytes != bytes || gpu.geom_bytes > EVICT_SMOKE_BUDGET / 100 * EVICT_HIGH_PCT {
            return Err(format!("Geometry bytes are off after eviction: {} tracked, {bytes} loaded", gpu.geom_bytes));
        }
        if gpu.model_count != n || gpu.instance_count != n || gpu.tri_count != n * 2 || gpu.batches.len() != n * 2 {
            return Err("Model counters are off after eviction".into());
        }
        // every model here brings two textures of its own, the uploads of evicted ones have to be gone
        let tex_bytes: usize = gpu.tex_uploads.values().map(|t| t.bytes).sum();
        if gpu.tex_uploads.len() != n * 2 || gpu.tex_bytes != tex_bytes {
            return Err("Texture uploads are off after eviction".into());
        }
        println!("WORLD_EVICTED {} of {} dropped, {n} models left", evicted.len(), dropped.len());
    } else if scene != "terrain" {
        let gpu = app.gpu.as_ref().ok_or("GPU did not initialize")?;
        if gpu.batches.len() != 2 {
            return Err("Per-material batching failed".into());
        }
        println!("WORLD_MATERIAL_BATCHES 2");
    }
    Ok(())
}

impl GpuState {
    // Smoke mode checks geometry coverage, not merely successful submission.
    pub(super) fn assert_visible_geometry(&self) {
        let row_bytes = (self.config.width * 4).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smoke_depth_readback"),
            size: row_bytes as u64 * self.config.height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.depth_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("Could not read GPU depth buffer");
        let data = buffer.slice(..).get_mapped_range();
        let mut visible = 0usize;
        let mut total = 0usize;
        // Exclude the minimap and window edges. Sky never writes depth.
        for y in self.config.height / 4..self.config.height * 3 / 4 {
            for x in self.config.width / 4..self.config.width * 3 / 4 {
                let offset = (y * row_bytes + x * 4) as usize;
                let depth = f32::from_ne_bytes(data[offset..offset + 4].try_into().unwrap());
                visible += usize::from(depth.is_finite() && depth < 1.0);
                total += 1;
            }
        }
        assert!(
            visible > total / 100,
            "Only sky is visible: geometry covers {visible}/{total} central pixels"
        );
        println!("WORLD_GEOMETRY_VISIBLE {visible}/{total}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    #[ignore = "requires cargo build --release and a GPU desktop"]
    fn gpu_world_launches_with_existing_event_loop() {
        use std::os::windows::process::CommandExt;
        use std::process::{Command, Stdio};
        use winit::platform::windows::EventLoopBuilderExtWindows;
        let _existing_event_loop = EventLoop::builder().with_any_thread(true).build().unwrap();
        assert!(matches!(
            EventLoop::builder().with_any_thread(true).build(),
            Err(winit::error::EventLoopError::RecreationAttempt)
        ));
        let executable = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("PercyStar.exe");
        for scene in ["terrain", "model", "stream", "evict"] {
            let mut child = Command::new(&executable)
                .args(["--gpu-world", "--smoke-scene", scene])
                .creation_flags(0x0800_0000)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let deadline = Instant::now() + std::time::Duration::from_secs(20);
            while child.try_wait().unwrap().is_none() {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let output = child.wait_with_output().unwrap();
                    panic!(
                        "{scene}: GPU smoke timed out: {}",
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{scene}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                stdout.contains("WORLD_GEOMETRY_VISIBLE"),
                "{scene}: no visible geometry"
            );
            if scene == "evict" {
                assert!(stdout.contains("WORLD_EVICTED"), "{scene}: nothing was evicted");
            } else if scene != "terrain" {
                assert!(stdout.contains("WORLD_MATERIAL_BATCHES 2"));
            }
        }
    }
}
