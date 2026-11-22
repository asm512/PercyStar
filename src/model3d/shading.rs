// shading normals for decoded meshes
// rage9 vertex normals are not decoded yet so the viewer and the exporters rebuild them from topology
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

// where the smoothing information of a mesh comes from
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum NormalSource {
    // generic triangle soup obj imports navmeshes terrain faces smooth only below the crease angle
    #[default]
    Crease,
    // gpu vertex buffers a vertex index shared by several faces always carried one normal
    // vertices duplicated at the same position are a uv seam or a hard edge and are told apart by angle
    SharedVertices,
}

// corners of position coincident vertices smooth together below these angles
const CREASE_COS: f32 = 0.766_044_4; // 40 degrees
const SEAM_COS: f32 = 0.642_787_6; // 50 degrees
// weld tolerance relative to the bounding box diagonal covers per geometry position quantization
const WELD_REL: f32 = 1.0e-4;
// position groups above this size fall back to a single average instead of pairwise tests
const PAIRWISE_LIMIT: usize = 256;

#[derive(Default)]
struct Mix64(u64);

impl Hasher for Mix64 {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn write_u64(&mut self, v: u64) {
        let x = v.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = x ^ (x >> 29);
    }
}

type CellMap = HashMap<u64, u32, BuildHasherDefault<Mix64>>;

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[inline]
fn unit(a: [f32; 3]) -> Option<[f32; 3]> {
    let l2 = dot(a, a);
    (l2.is_finite() && l2 > 1e-24).then(|| {
        let inv = l2.sqrt().recip();
        [a[0] * inv, a[1] * inv, a[2] * inv]
    })
}

fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let up = parent[parent[i as usize] as usize];
        parent[i as usize] = up;
        i = up;
    }
    i
}

// groups vertices that sit at the same position within the weld tolerance
// two half cell shifted grids so a pair straddling a cell border of one grid still meets in the other
fn position_classes(verts: &[[f32; 3]]) -> Vec<u32> {
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for v in verts {
        for k in 0..3 {
            if v[k].is_finite() {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
    }
    let diag = (0..3).map(|k| (hi[k] - lo[k]).max(0.0).powi(2)).sum::<f32>().sqrt();
    let cell = if diag.is_finite() && diag > 0.0 { diag * WELD_REL } else { 1.0 };
    let inv = 1.0 / cell;
    let mut parent: Vec<u32> = (0..verts.len() as u32).collect();
    for shift in [0.0f32, 0.5] {
        let mut cells = CellMap::with_capacity_and_hasher(verts.len(), Default::default());
        for (i, v) in verts.iter().enumerate() {
            let mut key = 0u64;
            for k in 0..3 {
                // saturating float cast keeps non finite input harmless
                let q = ((v[k] - lo[k]) * inv + shift) as i64;
                key = (key << 21) | (q.clamp(0, 0x1f_ffff) as u64);
            }
            match cells.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => {
                    let (a, b) = (find(&mut parent, *e.get()), find(&mut parent, i as u32));
                    if a != b {
                        parent[b.max(a) as usize] = b.min(a);
                    }
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(i as u32);
                }
            }
        }
    }
    (0..verts.len() as u32).map(|i| find(&mut parent, i)).collect()
}

// one smoothing participant weighted normal sum and its direction
struct Item {
    class: u32,
    sum: [f32; 3],
    dir: [f32; 3],
}

// replaces every item direction with the normalized sum of the compatible items in its position class
fn smooth_classes(items: &mut [Item], order: &mut [u32], cos_limit: f32) -> Vec<[f32; 3]> {
    order.sort_unstable_by_key(|&i| items[i as usize].class);
    let mut out = vec![[0.0f32; 3]; items.len()];
    let mut start = 0usize;
    while start < order.len() {
        let class = items[order[start] as usize].class;
        let mut end = start + 1;
        while end < order.len() && items[order[end] as usize].class == class {
            end += 1;
        }
        let run = &order[start..end];
        if run.len() == 1 {
            out[run[0] as usize] = items[run[0] as usize].dir;
        } else if run.len() > PAIRWISE_LIMIT {
            let mut total = [0.0f32; 3];
            for &j in run {
                let s = items[j as usize].sum;
                total = [total[0] + s[0], total[1] + s[1], total[2] + s[2]];
            }
            let avg = unit(total);
            for &i in run {
                let d = items[i as usize].dir;
                out[i as usize] = match avg {
                    Some(a) if dot(a, d) >= cos_limit => a,
                    _ => d,
                };
            }
        } else {
            for &i in run {
                let d = items[i as usize].dir;
                let mut acc = [0.0f32; 3];
                for &j in run {
                    let o = &items[j as usize];
                    if dot(d, o.dir) >= cos_limit {
                        acc = [acc[0] + o.sum[0], acc[1] + o.sum[1], acc[2] + o.sum[2]];
                    }
                }
                out[i as usize] = unit(acc).unwrap_or(d);
            }
        }
        start = end;
    }
    out
}

// interior angle of every corner used as the smoothing weight so the result does not depend on tessellation
fn corner_angles(verts: &[[f32; 3]], t: [u32; 3]) -> [f32; 3] {
    let p = [verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize]];
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let (a, b) = (sub(p[(c + 1) % 3], p[c]), sub(p[(c + 2) % 3], p[c]));
        let x = cross(a, b);
        let sin = dot(x, x).sqrt();
        // a zero area corner has no normal to contribute and atan2 of signed zeros would report pi
        out[c] = if sin > 0.0 { sin.atan2(dot(a, b)) } else { 0.0 };
    }
    out
}

// unit shading normal for the three corners of every triangle
// face_normals are the unit geometric normals parallel to tris and every index must be inside verts
pub fn corner_normals(verts: &[[f32; 3]], tris: &[[u32; 3]], face_normals: &[[f32; 3]], source: NormalSource) -> Vec<[[f32; 3]; 3]> {
    if tris.is_empty() || face_normals.len() != tris.len() {
        return Vec::new();
    }
    let classes = position_classes(verts);
    match source {
        NormalSource::SharedVertices => {
            let mut acc = vec![[0.0f32; 3]; verts.len()];
            let mut used = vec![false; verts.len()];
            for (t, n) in tris.iter().zip(face_normals) {
                let w = corner_angles(verts, *t);
                for c in 0..3 {
                    let a = &mut acc[t[c] as usize];
                    *a = [a[0] + n[0] * w[c], a[1] + n[1] * w[c], a[2] + n[2] * w[c]];
                    used[t[c] as usize] = true;
                }
            }
            // item index equals vertex index unused vertices keep a zero direction and are never read
            let mut items: Vec<Item> = acc
                .iter()
                .zip(&classes)
                .map(|(&sum, &class)| Item { class, sum, dir: unit(sum).unwrap_or([0.0; 3]) })
                .collect();
            let mut order: Vec<u32> = (0..verts.len() as u32).filter(|&i| used[i as usize]).collect();
            let smooth = smooth_classes(&mut items, &mut order, SEAM_COS);
            tris.iter()
                .zip(face_normals)
                .map(|(t, &n)| std::array::from_fn(|c| unit(smooth[t[c] as usize]).unwrap_or(n)))
                .collect()
        }
        NormalSource::Crease => {
            let mut items = Vec::with_capacity(tris.len() * 3);
            for (t, &n) in tris.iter().zip(face_normals) {
                let w = corner_angles(verts, *t);
                for c in 0..3 {
                    items.push(Item { class: classes[t[c] as usize], sum: [n[0] * w[c], n[1] * w[c], n[2] * w[c]], dir: n });
                }
            }
            let mut order: Vec<u32> = (0..items.len() as u32).collect();
            let smooth = smooth_classes(&mut items, &mut order, CREASE_COS);
            smooth.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face_normals(verts: &[[f32; 3]], tris: &[[u32; 3]]) -> Vec<[f32; 3]> {
        tris.iter()
            .map(|t| {
                let (a, b, c) = (verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize]);
                unit(cross(sub(b, a), sub(c, a))).unwrap_or([0.0, 0.0, 1.0])
            })
            .collect()
    }

    // uv sphere with shared indices except along one duplicated seam column
    fn sphere(segs: u32, rings: u32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
        let mut verts = Vec::new();
        for r in 0..=rings {
            let phi = std::f32::consts::PI * r as f32 / rings as f32;
            for s in 0..=segs {
                let th = std::f32::consts::TAU * s as f32 / segs as f32;
                verts.push([phi.sin() * th.cos(), phi.sin() * th.sin(), phi.cos()]);
            }
        }
        let mut tris = Vec::new();
        for r in 0..rings {
            for s in 0..segs {
                let a = r * (segs + 1) + s;
                let b = a + segs + 1;
                if r > 0 {
                    tris.push([a, b, a + 1]);
                }
                if r + 1 < rings {
                    tris.push([a + 1, b, b + 1]);
                }
            }
        }
        (verts, tris)
    }

    #[test]
    fn sphere_normals_follow_the_surface_across_the_duplicated_seam() {
        let (verts, tris) = sphere(24, 12);
        let fnorm = face_normals(&verts, &tris);
        for source in [NormalSource::SharedVertices, NormalSource::Crease] {
            let cn = corner_normals(&verts, &tris, &fnorm, source);
            assert_eq!(cn.len(), tris.len());
            let mut worst = 1.0f32;
            for (t, n) in tris.iter().zip(&cn) {
                for c in 0..3 {
                    // on a unit sphere the true normal is the position itself
                    worst = worst.min(dot(n[c], verts[t[c] as usize]));
                    assert!((dot(n[c], n[c]) - 1.0).abs() < 1e-4, "normals are unit length");
                }
            }
            assert!(worst > 0.985, "{source:?} smooth normals deviate from the sphere: {worst}");
            // the seam column is duplicated geometry both copies must agree
            let seam: Vec<_> = tris
                .iter()
                .zip(&cn)
                .flat_map(|(t, n)| (0..3).map(move |c| (t[c], n[c])))
                .filter(|(i, _)| i % 25 == 0 || i % 25 == 24)
                .collect();
            for (i, n) in &seam {
                let twin = if i % 25 == 0 { i + 24 } else { i - 24 };
                for (j, m) in &seam {
                    if *j == twin {
                        assert!(dot(*n, *m) > 0.999, "{source:?} seam vertex {i} and its duplicate {j} shade differently");
                    }
                }
            }
        }
    }


    #[test]
    fn cube_keeps_hard_edges_when_faces_do_not_share_vertices() {
        // 24 vertex cube the way a gpu vertex buffer stores hard edges
        let mut verts = Vec::new();
        let mut tris = Vec::new();
        for axis in 0..3 {
            for sign in [-1.0f32, 1.0] {
                let base = verts.len() as u32;
                for (u, v) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    let mut p = [0.0f32; 3];
                    p[axis] = sign;
                    p[(axis + 1) % 3] = u;
                    p[(axis + 2) % 3] = v;
                    verts.push(p);
                }
                tris.push(if sign > 0.0 { [base, base + 1, base + 2] } else { [base, base + 2, base + 1] });
                tris.push(if sign > 0.0 { [base, base + 2, base + 3] } else { [base, base + 3, base + 2] });
            }
        }
        let fnorm = face_normals(&verts, &tris);
        for source in [NormalSource::SharedVertices, NormalSource::Crease] {
            let cn = corner_normals(&verts, &tris, &fnorm, source);
            for (n, f) in cn.iter().zip(&fnorm) {
                for corner in n {
                    assert!(dot(*corner, *f) > 0.9999, "{source:?} rounded a 90 degree edge");
                }
            }
        }
    }

    #[test]
    fn shared_index_cube_is_faceted_as_a_soup_and_rounded_as_a_vertex_buffer() {
        let verts: Vec<[f32; 3]> = (0..8).map(|i| [(i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32]).collect();
        let tris = vec![
            [0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
            [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5],
        ];
        let fnorm = face_normals(&verts, &tris);
        let soup = corner_normals(&verts, &tris, &fnorm, NormalSource::Crease);
        for (n, f) in soup.iter().zip(&fnorm) {
            assert!((0..3).all(|c| dot(n[c], *f) > 0.9999), "crease mode must not round a cube");
        }
        let vb = corner_normals(&verts, &tris, &fnorm, NormalSource::SharedVertices);
        let rounded = vb.iter().zip(&fnorm).filter(|(n, f)| (0..3).any(|c| dot(n[c], **f) < 0.9)).count();
        assert_eq!(rounded, tris.len(), "a shared vertex carries one normal for all of its faces");
    }

    #[test]
    fn quantization_noise_between_geometries_still_welds() {
        // two coplanar quads meeting at x 1 the second one offset by a 16 bit quantization step
        let e = 1.5e-5;
        let verts = vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.3], [0.0, 1.0, 0.3],
            [1.0, 0.0, e], [2.0, 0.0, 0.3], [2.0, 1.0, 0.6], [1.0, 1.0, 0.3 + e],
        ];
        let tris = vec![[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]];
        let fnorm = face_normals(&verts, &tris);
        let cn = corner_normals(&verts, &tris, &fnorm, NormalSource::SharedVertices);
        // vertex 1 of the first quad and vertex 4 of the second sit on the shared edge
        assert!(dot(cn[0][1], cn[2][0]) > 0.9999, "seam vertices must share one normal");
        assert!(dot(cn[0][1], fnorm[0]) < 0.9999, "seam normal blends both quads");
    }

    #[test]
    fn zero_area_triangles_do_not_pollute_their_neighbours() {
        // pole fan closed with collapsed triangles whose fallback normal points the other way
        let n = 12u32;
        // shallow cone so every face stays inside the seam angle of the others
        let mut verts = vec![[0.0f32, 0.0, 0.2]; n as usize];
        for i in 0..n {
            let a = std::f32::consts::TAU * i as f32 / n as f32;
            verts.push([a.cos(), a.sin(), 0.0]);
        }
        let mut tris = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            tris.push([i, n + i, n + j]);
            // collapsed triangle two corners on the pole with negative zero edge components
            verts[i as usize][1] = if i % 2 == 0 { 0.0 } else { -0.0 };
            tris.push([i, j, n + j]);
        }
        let mut fnorm = face_normals(&verts, &tris);
        for (t, f) in tris.iter().zip(&mut fnorm) {
            if t[0] < n && t[1] < n {
                *f = [0.0, 0.0, -1.0];
            }
        }
        let cn = corner_normals(&verts, &tris, &fnorm, NormalSource::SharedVertices);
        for (t, c) in tris.iter().zip(&cn) {
            if t[1] >= n {
                assert!(c[0][2] > 0.999 && c[0][0].abs() < 1e-3 && c[0][1].abs() < 1e-3, "pole normal {:?}", c[0]);
            }
        }
    }

    #[test]
    fn degenerate_and_empty_input_is_harmless() {
        assert!(corner_normals(&[], &[], &[], NormalSource::Crease).is_empty());
        let verts = vec![[0.0; 3], [0.0; 3], [f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let tris = vec![[0, 1, 2], [0, 3, 4], [0, 0, 0]];
        let fnorm = vec![[0.0, 0.0, 1.0]; 3];
        for source in [NormalSource::SharedVertices, NormalSource::Crease] {
            let cn = corner_normals(&verts, &tris, &fnorm, source);
            assert_eq!(cn.len(), 3);
            assert!(cn.iter().flatten().flatten().all(|v| v.is_finite()), "{source:?} produced a non finite normal");
        }
    }
}
