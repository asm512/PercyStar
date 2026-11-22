// mesh exporters wavefront obj with normals and materials and binary gltf
use super::texture::srgb_to_linear_table;
use super::{GeomMaterial, MaterialPass, Mesh3D, Texture2D, GLASS_PREVIEW_ALPHA};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

const DEFAULT_COLOR: [f32; 3] = [0.8, 0.8, 0.8];

// file name safe ascii stem
fn safe_stem(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '-') { c } else { '_' }).collect();
    let s = s.trim_matches('_').to_string();
    if s.is_empty() { "model".into() } else { s }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// png bytes of a model texture opaque materials drop the alpha channel because it may carry unrelated data
fn encode_png(tex: &Texture2D, keep_alpha: bool) -> std::io::Result<Vec<u8>> {
    let mut out = std::io::Cursor::new(Vec::new());
    let (bytes, kind): (Vec<u8>, _) = if keep_alpha {
        (tex.pixels.iter().flat_map(|c| c.to_array()).collect(), image::ExtendedColorType::Rgba8)
    } else {
        (tex.pixels.iter().flat_map(|c| [c.r(), c.g(), c.b()]).collect(), image::ExtendedColorType::Rgb8)
    };
    image::write_buffer_with_format(&mut out, &bytes, tex.w as u32, tex.h as u32, kind, image::ImageFormat::Png)
        .map_err(std::io::Error::other)?;
    Ok(out.into_inner())
}

// what an exported material looks like resolved from the preview material of one geometry
struct MatInfo<'a> {
    name: String,
    // srgb base color and opacity
    color: [f32; 3],
    opacity: f32,
    texture: Option<&'a Arc<Texture2D>>,
    normal: Option<&'a Arc<Texture2D>>,
    pass: MaterialPass,
    paint: bool,
    auxiliary: bool,
}

impl Mesh3D {
    fn export_material(&self, g: usize) -> MatInfo<'_> {
        let m: Option<&GeomMaterial> = self.geom_materials.get(g);
        let pass = m.map_or(MaterialPass::Opaque, |m| m.pass);
        let glass = matches!(pass, MaterialPass::Glass { .. });
        let paint = m.is_some_and(|m| m.is_vehicle_paint);
        let auxiliary = m.is_some_and(|m| m.auxiliary_pass);
        MatInfo {
            name: format!("geom_{g:03}{}", if auxiliary { "_aux" } else if paint { "_paint" } else if glass { "_glass" } else { "" }),
            color: if paint { self.paint_color } else if glass { [0.62, 0.71, 0.78] } else { DEFAULT_COLOR },
            // glass that carries its own coverage in the texture is not dimmed twice
            opacity: if matches!(pass, MaterialPass::Glass { texture_alpha: false }) || (glass && m.is_some_and(|m| m.texture.is_none())) { GLASS_PREVIEW_ALPHA as f32 / 255.0 } else { 1.0 },
            texture: m.and_then(|m| m.texture.as_ref()),
            normal: m.and_then(|m| m.normal.as_ref()),
            pass,
            paint,
            auxiliary,
        }
    }

    // triangle ranges to export with their group name the isolated part or every part
    fn export_groups(&self) -> Vec<(&str, std::ops::Range<usize>)> {
        match self.active_part.and_then(|i| self.parts.get(i)) {
            Some(p) => vec![(p.name.as_str(), self.visible_tris())],
            None if !self.parts.is_empty() => self.parts.iter().map(|p| (p.name.as_str(), p.tris.clone())).collect(),
            None => vec![(self.name.as_str(), 0..self.tris.len())],
        }
    }

    // writes visible parts as obj one object per drawable with smooth normals and recenters an isolated part
    pub fn export_obj(&self, path: &Path) -> std::io::Result<()> {
        self.write_obj(path, false)
    }

    // obj plus a material library and the applied textures as png next to it
    pub fn export_obj_with_materials(&self, path: &Path) -> std::io::Result<()> {
        self.write_obj(path, true)
    }

    fn write_obj(&self, path: &Path, materials: bool) -> std::io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "# PercyStar RAGE9 Model Export: {}", self.name)?;
        writeln!(w, "# z up, vt origin bottom left")?;
        let stem = safe_stem(&path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default());
        let groups = self.export_groups();
        let materials = materials && !self.tri_geom.is_empty();
        if materials {
            writeln!(w, "mtllib {stem}.mtl")?;
        }
        let shift = if self.active_part.is_some() { [self.center[0], self.center[1], 0.0] } else { [0.0; 3] };
        let has_uv = self.uvs.len() == self.verts.len() && !self.uvs.is_empty();
        let normals = self.corner_normals();
        let has_n = normals.len() == self.tris.len();
        let mut remap = vec![0u32; self.verts.len()];
        let mut next = 1u32;
        let mut normal_ids: HashMap<[i32; 3], u32> = HashMap::new();
        let mut used_geoms: Vec<usize> = Vec::new();
        for (name, range) in groups {
            writeln!(w, "o {name}")?;
            let tris = &self.tris[range.clone()];
            for &i in tris.iter().flatten() {
                let i = i as usize;
                if remap[i] == 0 {
                    remap[i] = next;
                    next += 1;
                    let v = self.verts[i];
                    writeln!(w, "v {:.6} {:.6} {:.6}", v[0] - shift[0], v[1] - shift[1], v[2] - shift[2])?;
                    if has_uv {
                        // textures are stored top row first obj counts v from the bottom
                        writeln!(w, "vt {:.6} {:.6}", self.uvs[i][0], 1.0 - self.uvs[i][1])?;
                    }
                }
            }
            // normals are written once and shared between the faces that use them
            let mut tri_normals: Vec<[u32; 3]> = Vec::new();
            if has_n {
                tri_normals.reserve(tris.len());
                for ti in range.clone() {
                    let mut ids = [0u32; 3];
                    for c in 0..3 {
                        let n = normals[ti][c];
                        let key = n.map(|v| (v * 32767.0).round() as i32);
                        let fresh = normal_ids.len() as u32 + 1;
                        let id = *normal_ids.entry(key).or_insert(fresh);
                        if id == fresh {
                            writeln!(w, "vn {:.5} {:.5} {:.5}", n[0], n[1], n[2])?;
                        }
                        ids[c] = id;
                    }
                    tri_normals.push(ids);
                }
            }
            let mut current: Option<usize> = None;
            for (k, t) in tris.iter().enumerate() {
                let g = self.tri_geom.get(range.start + k).copied().unwrap_or(0) as usize;
                if materials && current != Some(g) {
                    current = Some(g);
                    if !used_geoms.contains(&g) {
                        used_geoms.push(g);
                    }
                    writeln!(w, "usemtl {}", self.export_material(g).name)?;
                }
                let v = [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]];
                match (has_uv, tri_normals.get(k)) {
                    (true, Some(n)) => writeln!(w, "f {0}/{0}/{3} {1}/{1}/{4} {2}/{2}/{5}", v[0], v[1], v[2], n[0], n[1], n[2])?,
                    (false, Some(n)) => writeln!(w, "f {0}//{3} {1}//{4} {2}//{5}", v[0], v[1], v[2], n[0], n[1], n[2])?,
                    (true, None) => writeln!(w, "f {0}/{0} {1}/{1} {2}/{2}", v[0], v[1], v[2])?,
                    (false, None) => writeln!(w, "f {} {} {}", v[0], v[1], v[2])?,
                }
            }
        }
        w.flush()?;
        if !materials {
            return Ok(());
        }

        let dir = path.parent().unwrap_or(Path::new("."));
        let mut mtl = BufWriter::new(File::create(dir.join(format!("{stem}.mtl")))?);
        writeln!(mtl, "# PercyStar material library for {}", self.name)?;
        // every distinct texture is written once
        let mut written: Vec<(*const Texture2D, String)> = Vec::new();
        let mut image_for = |tex: &Arc<Texture2D>, keep_alpha: bool, tag: &str| -> std::io::Result<String> {
            let ptr = Arc::as_ptr(tex);
            if let Some((_, name)) = written.iter().find(|(p, _)| *p == ptr) {
                return Ok(name.clone());
            }
            let name = format!("{stem}_{tag}{:02}.png", written.len());
            std::fs::write(dir.join(&name), encode_png(tex, keep_alpha)?)?;
            written.push((ptr, name.clone()));
            Ok(name)
        };
        used_geoms.sort_unstable();
        for g in used_geoms {
            let m = self.export_material(g);
            writeln!(mtl, "\nnewmtl {}", m.name)?;
            writeln!(mtl, "Kd {:.4} {:.4} {:.4}", m.color[0], m.color[1], m.color[2])?;
            writeln!(mtl, "Ka 0 0 0")?;
            writeln!(mtl, "Ks {0} {0} {0}", if m.paint { "0.5" } else { "0.04" })?;
            writeln!(mtl, "Ns {}", if m.paint { 250 } else { 40 })?;
            writeln!(mtl, "d {:.3}", m.opacity)?;
            writeln!(mtl, "illum 2")?;
            if let (Some(tex), true) = (m.texture, has_uv) {
                let uses_alpha = m.pass != MaterialPass::Opaque && tex.has_alpha();
                let name = image_for(tex, uses_alpha, "color")?;
                writeln!(mtl, "map_Kd {name}")?;
                if uses_alpha {
                    writeln!(mtl, "map_d {name}")?;
                }
            }
            if let (Some(tex), true) = (m.normal, has_uv) {
                writeln!(mtl, "norm {}", image_for(tex, false, "normal")?)?;
            }
        }
        mtl.flush()
    }

    // binary gltf 2 y up one node per drawable one primitive per material textures embedded as png
    // shadow proxy geometry is left out
    pub fn export_glb(&self, path: &Path) -> std::io::Result<()> {
        let to_linear = srgb_to_linear_table();
        let lin = |c: f32| to_linear[(c.clamp(0.0, 1.0) * 255.0 + 0.5) as usize];
        let has_uv = self.uvs.len() == self.verts.len() && !self.uvs.is_empty();
        let normals = self.corner_normals();
        let has_n = normals.len() == self.tris.len();
        let shift = if self.active_part.is_some() { [self.center[0], self.center[1], 0.0] } else { [0.0; 3] };
        // z up to gltf y up
        let y_up = |v: [f32; 3]| [v[0], v[2], -v[1]];

        let mut bin: Vec<u8> = Vec::new();
        let mut views: Vec<String> = Vec::new();
        let mut accessors: Vec<String> = Vec::new();
        let mut images: Vec<String> = Vec::new();
        let mut textures: Vec<String> = Vec::new();
        let mut materials: Vec<String> = Vec::new();
        let mut meshes: Vec<String> = Vec::new();
        let mut nodes: Vec<String> = Vec::new();
        let mut image_ids: Vec<(*const Texture2D, bool, usize)> = Vec::new();
        let mut material_ids: HashMap<usize, usize> = HashMap::new();

        fn push_view(bin: &mut Vec<u8>, views: &mut Vec<String>, data: &[u8], target: Option<u32>) -> usize {
            while !bin.len().is_multiple_of(4) {
                bin.push(0);
            }
            let target = target.map(|t| format!(",\"target\":{t}")).unwrap_or_default();
            views.push(format!("{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{}{target}}}", bin.len(), data.len()));
            bin.extend_from_slice(data);
            views.len() - 1
        }

        for (name, range) in self.export_groups() {
            // split the group by geometry keeping first use order
            let mut by_geom: Vec<(usize, Vec<usize>)> = Vec::new();
            for ti in range {
                let g = self.tri_geom.get(ti).copied().unwrap_or(0) as usize;
                match by_geom.iter_mut().find(|(gg, _)| *gg == g) {
                    Some((_, list)) => list.push(ti),
                    None => by_geom.push((g, vec![ti])),
                }
            }
            let mut primitives: Vec<String> = Vec::new();
            for (g, tri_ids) in by_geom {
                let info = self.export_material(g);
                if info.auxiliary {
                    continue;
                }
                // one gltf vertex per source vertex and shading normal
                let mut lookup: HashMap<(u32, [i16; 3]), u32> = HashMap::new();
                let (mut pos, mut nrm, mut uv, mut idx): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
                for ti in tri_ids {
                    for (c, &vi) in self.tris[ti].iter().enumerate() {
                        let n = if has_n { normals[ti][c] } else { self.normals.get(ti).copied().unwrap_or([0.0, 0.0, 1.0]) };
                        let key = (vi, n.map(|v| (v * 32767.0).round() as i16));
                        let fresh = (pos.len() / 3) as u32;
                        let id = *lookup.entry(key).or_insert(fresh);
                        if id == fresh {
                            let v = self.verts[vi as usize];
                            let p = y_up([v[0] - shift[0], v[1] - shift[1], v[2] - shift[2]]);
                            for k in 0..3 {
                                lo[k] = lo[k].min(p[k]);
                                hi[k] = hi[k].max(p[k]);
                            }
                            pos.extend(p);
                            // gltf requires unit normals
                            let m = y_up(n);
                            let len = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
                            nrm.extend(if len > 1e-6 { [m[0] / len, m[1] / len, m[2] / len] } else { [0.0, 1.0, 0.0] });
                            if has_uv {
                                uv.extend(self.uvs[vi as usize]);
                            }
                        }
                        idx.push(id);
                    }
                }
                if idx.is_empty() || !lo.iter().chain(&hi).all(|v| v.is_finite()) {
                    continue;
                }
                let count = pos.len() / 3;
                let bytes = |f: &[f32]| f.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
                let pv = push_view(&mut bin, &mut views, &bytes(&pos), Some(34962));
                accessors.push(format!(
                    "{{\"bufferView\":{pv},\"componentType\":5126,\"count\":{count},\"type\":\"VEC3\",\"min\":[{},{},{}],\"max\":[{},{},{}]}}",
                    lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
                ));
                let mut attrs = format!("\"POSITION\":{}", accessors.len() - 1);
                let nv = push_view(&mut bin, &mut views, &bytes(&nrm), Some(34962));
                accessors.push(format!("{{\"bufferView\":{nv},\"componentType\":5126,\"count\":{count},\"type\":\"VEC3\"}}"));
                attrs += &format!(",\"NORMAL\":{}", accessors.len() - 1);
                if has_uv {
                    let tv = push_view(&mut bin, &mut views, &bytes(&uv), Some(34962));
                    accessors.push(format!("{{\"bufferView\":{tv},\"componentType\":5126,\"count\":{count},\"type\":\"VEC2\"}}"));
                    attrs += &format!(",\"TEXCOORD_0\":{}", accessors.len() - 1);
                }
                let ib: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
                let iv = push_view(&mut bin, &mut views, &ib, Some(34963));
                accessors.push(format!("{{\"bufferView\":{iv},\"componentType\":5125,\"count\":{},\"type\":\"SCALAR\"}}", idx.len()));
                let index_accessor = accessors.len() - 1;

                let material = match material_ids.get(&g) {
                    Some(&m) => m,
                    None => {
                        let mut texture_index = |tex: &Arc<Texture2D>, keep_alpha: bool| -> std::io::Result<usize> {
                            let ptr = Arc::as_ptr(tex);
                            if let Some((_, _, i)) = image_ids.iter().find(|(p, a, _)| *p == ptr && *a == keep_alpha) {
                                return Ok(*i);
                            }
                            let view = push_view(&mut bin, &mut views, &encode_png(tex, keep_alpha)?, None);
                            images.push(format!("{{\"bufferView\":{view},\"mimeType\":\"image/png\"}}"));
                            textures.push(format!("{{\"sampler\":0,\"source\":{}}}", images.len() - 1));
                            image_ids.push((ptr, keep_alpha, textures.len() - 1));
                            Ok(textures.len() - 1)
                        };
                        let textured = info.texture.filter(|_| has_uv);
                        let uses_alpha = info.pass != MaterialPass::Opaque && textured.is_some_and(|t| t.has_alpha());
                        // a texture replaces the flat preview color
                        let base = if textured.is_some() { [1.0, 1.0, 1.0] } else { info.color.map(lin) };
                        let mut pbr = format!("\"baseColorFactor\":[{},{},{},{}]", base[0], base[1], base[2], info.opacity);
                        if let Some(tex) = textured {
                            pbr += &format!(",\"baseColorTexture\":{{\"index\":{}}}", texture_index(tex, uses_alpha)?);
                        }
                        let (metal, rough) = if info.paint { (0.35, 0.3) } else if matches!(info.pass, MaterialPass::Glass { .. }) { (0.0, 0.08) } else { (0.0, 0.7) };
                        pbr += &format!(",\"metallicFactor\":{metal},\"roughnessFactor\":{rough}");
                        let mut mat = format!("{{\"name\":{},\"pbrMetallicRoughness\":{{{pbr}}},\"doubleSided\":true", json_str(&info.name));
                        if let Some(tex) = info.normal.filter(|_| has_uv) {
                            mat += &format!(",\"normalTexture\":{{\"index\":{}}}", texture_index(tex, false)?);
                        }
                        mat += match info.pass {
                            MaterialPass::Cutout if uses_alpha => ",\"alphaMode\":\"MASK\",\"alphaCutoff\":0.5",
                            MaterialPass::Glass { .. } => ",\"alphaMode\":\"BLEND\"",
                            MaterialPass::Alpha | MaterialPass::Decal if uses_alpha => ",\"alphaMode\":\"BLEND\"",
                            _ => "",
                        };
                        mat.push('}');
                        materials.push(mat);
                        material_ids.insert(g, materials.len() - 1);
                        materials.len() - 1
                    }
                };
                primitives.push(format!("{{\"attributes\":{{{attrs}}},\"indices\":{index_accessor},\"material\":{material}}}"));
            }
            if primitives.is_empty() {
                continue;
            }
            meshes.push(format!("{{\"name\":{},\"primitives\":[{}]}}", json_str(name), primitives.join(",")));
            nodes.push(format!("{{\"name\":{},\"mesh\":{}}}", json_str(name), meshes.len() - 1));
        }
        if meshes.is_empty() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "mesh has no triangles to export"));
        }
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }

        let mut json = format!(
            "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"PercyStar {}\"}},\"scene\":0,\"scenes\":[{{\"name\":{},\"nodes\":[{}]}}]",
            env!("CARGO_PKG_VERSION"),
            json_str(&self.name),
            (0..nodes.len()).map(|i| i.to_string()).collect::<Vec<_>>().join(",")
        );
        json += &format!(",\"nodes\":[{}],\"meshes\":[{}]", nodes.join(","), meshes.join(","));
        json += &format!(",\"materials\":[{}]", materials.join(","));
        if !textures.is_empty() {
            // repeat addressing with trilinear filtering
            json += &format!(",\"textures\":[{}],\"images\":[{}]", textures.join(","), images.join(","));
            json += ",\"samplers\":[{\"magFilter\":9729,\"minFilter\":9987,\"wrapS\":10497,\"wrapT\":10497}]";
        }
        json += &format!(",\"accessors\":[{}],\"bufferViews\":[{}]", accessors.join(","), views.join(","));
        json += &format!(",\"buffers\":[{{\"byteLength\":{}}}]}}", bin.len());
        let mut json = json.into_bytes();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let total = 12 + 8 + json.len() + 8 + bin.len();
        let total = u32::try_from(total).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "model exceeds the 4 GiB glb limit"))?;
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"glTF")?;
        w.write_all(&2u32.to_le_bytes())?;
        w.write_all(&total.to_le_bytes())?;
        w.write_all(&(json.len() as u32).to_le_bytes())?;
        w.write_all(b"JSON")?;
        w.write_all(&json)?;
        w.write_all(&(bin.len() as u32).to_le_bytes())?;
        w.write_all(b"BIN\0")?;
        w.write_all(&bin)?;
        w.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("percystar_export_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn obj_export_writes_normals_flipped_v_and_materials() {
        let mesh = crate::model3d::showcase_tests::showcase();
        let dir = temp_dir("obj");
        let path = dir.join("show case#1.obj");
        mesh.export_obj_with_materials(&path).unwrap();
        let obj = std::fs::read_to_string(&path).unwrap();
        assert!(obj.starts_with("# PercyStar"), "header keeps the z up marker parse_obj relies on");
        assert!(obj.contains("\nmtllib show_case_1.mtl\n"));
        let count = |p: &str| obj.lines().filter(|l| l.starts_with(p)).count();
        assert_eq!(count("v "), mesh.verts.len());
        assert_eq!(count("vt "), mesh.verts.len());
        assert_eq!(count("f "), mesh.tris.len());
        assert!(count("vn ") > 100 && count("vn ") < mesh.tris.len() * 3, "normals are shared: {}", count("vn "));
        assert_eq!(count("usemtl "), 7);
        // every face references position uv and normal and every index is in range
        let (nv, nn) = (count("v ") as u32, count("vn ") as u32);
        for l in obj.lines().filter(|l| l.starts_with("f ")) {
            for tok in l[2..].split(' ') {
                let ids: Vec<u32> = tok.split('/').map(|s| s.parse().unwrap()).collect();
                assert!(ids.len() == 3 && ids[0] >= 1 && ids[0] <= nv && ids[1] == ids[0] && ids[2] >= 1 && ids[2] <= nn, "{l}");
            }
        }
        // first vertex has v 0 in the mesh and 1 in the file
        assert_eq!(mesh.uvs[0][1], 0.0);
        assert!(obj.lines().find(|l| l.starts_with("vt ")).unwrap().ends_with(" 1.000000"));
        let mtl = std::fs::read_to_string(dir.join("show_case_1.mtl")).unwrap();
        assert_eq!(mtl.matches("newmtl ").count(), 7);
        assert!(mtl.contains("newmtl geom_000_paint") && mtl.contains("newmtl geom_006_glass") && mtl.contains("map_d "));
        for line in mtl.lines().filter(|l| l.starts_with("map_Kd ")) {
            let img = image::open(dir.join(&line[7..])).expect("texture written next to the obj");
            assert!(img.width() > 0);
        }
        // the plain exporter stays a single file without materials and loads back
        let plain = dir.join("plain.obj");
        mesh.export_obj(&plain).unwrap();
        let text = std::fs::read_to_string(&plain).unwrap();
        assert!(!text.contains("mtllib") && !text.contains("usemtl") && text.contains("\nvn "));
        let back = crate::model3d::parse_obj(&plain).expect("reload");
        assert_eq!((back.verts.len(), back.tris.len()), (mesh.verts.len(), mesh.tris.len()));
        // percystar files keep their z up axes so the bounds match without a flip
        assert!((0..3).all(|k| (back.center[k] - mesh.center[k]).abs() < 1e-4) && (back.radius - mesh.radius).abs() < 1e-4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn glb_export_is_a_well_formed_container() {
        let mut mesh = crate::model3d::showcase_tests::showcase();
        mesh.name = "quote\"and\\slash".into();
        let dir = temp_dir("glb");
        let path = dir.join("showcase.glb");
        mesh.export_glb(&path).unwrap();
        let data = std::fs::read(&path).unwrap();
        let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as usize;
        assert_eq!(&data[..4], b"glTF");
        assert_eq!((u32_at(4), u32_at(8)), (2, data.len()));
        let json_len = u32_at(12);
        assert_eq!(&data[16..20], b"JSON");
        assert!(json_len.is_multiple_of(4));
        let json = std::str::from_utf8(&data[20..20 + json_len]).unwrap();
        let bin_at = 20 + json_len;
        let bin_len = u32_at(bin_at);
        assert_eq!(&data[bin_at + 4..bin_at + 8], b"BIN\0");
        assert_eq!(bin_at + 8 + bin_len, data.len());
        assert!(json.contains(&format!("\"buffers\":[{{\"byteLength\":{bin_len}}}]")));
        // balanced structure and escaped strings
        let (mut depth, mut in_str, mut esc) = (0i32, false, false);
        for c in json.chars() {
            if in_str {
                if esc { esc = false } else if c == '\\' { esc = true } else if c == '"' { in_str = false }
            } else {
                match c {
                    '"' => in_str = true,
                    '{' | '[' => depth += 1,
                    '}' | ']' => depth -= 1,
                    _ => {}
                }
                assert!(depth >= 0);
            }
        }
        assert!(depth == 0 && !in_str);
        assert!(json.contains("quote\\\"and\\\\slash"));
        assert!(!json.contains("NaN") && !json.contains("inf"));
        // seven materials five textures for four textured geometries would mean a duplicate
        assert_eq!(json.matches("\"pbrMetallicRoughness\"").count(), 7);
        assert_eq!(json.matches("\"mimeType\":\"image/png\"").count(), 4);
        assert!(json.contains("\"alphaMode\":\"MASK\"") && json.contains("\"alphaMode\":\"BLEND\""));
        // every buffer view lies inside the binary chunk and is aligned
        for part in json.split("\"byteOffset\":").skip(1) {
            let off: usize = part.split(',').next().unwrap().parse().unwrap();
            let len: usize = part.split("\"byteLength\":").nth(1).unwrap().split([',', '}']).next().unwrap().parse().unwrap();
            assert!(off.is_multiple_of(4) && off + len <= bin_len);
        }
        // shadow proxies are skipped and an empty selection is an error instead of an empty file
        let mut hidden = mesh.clone();
        for m in &mut hidden.geom_materials {
            m.auxiliary_pass = true;
        }
        assert!(hidden.export_glb(&dir.join("none.glb")).is_err());
        if let Ok(keep) = std::env::var("PERCYSTAR_RENDER_DIR") {
            let _ = std::fs::create_dir_all(&keep);
            std::fs::copy(&path, Path::new(&keep).join("showcase.glb")).unwrap();
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
