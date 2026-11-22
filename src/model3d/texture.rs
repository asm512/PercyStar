// model viewer textures filtered sampling and the srgb transfer tables
use eframe::egui::Color32;
use std::sync::OnceLock;

struct Tables {
    to_linear: [f32; 256],
    to_srgb: [u8; 4097],
}

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut to_linear = [0.0f32; 256];
        for (i, v) in to_linear.iter_mut().enumerate() {
            let c = i as f32 / 255.0;
            *v = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
        }
        let mut to_srgb = [0u8; 4097];
        for (i, v) in to_srgb.iter_mut().enumerate() {
            let l = i as f32 / 4096.0;
            let c = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
            *v = (c * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
        Tables { to_linear, to_srgb }
    })
}

// srgb byte to linear light
#[inline]
pub fn srgb_to_linear_table() -> &'static [f32; 256] {
    &tables().to_linear
}

// linear light 0 to 1 to srgb byte index with linear * 4096
#[inline]
pub fn linear_to_srgb_table() -> &'static [u8; 4097] {
    &tables().to_srgb
}

#[inline]
pub fn encode_srgb(table: &[u8; 4097], linear: f32) -> u8 {
    // the float to int cast saturates and maps nan to zero
    table[((linear * 4096.0 + 0.5) as usize).min(4096)]
}

#[derive(Clone)]
struct MipLevel {
    w: usize,
    h: usize,
    pixels: Vec<Color32>,
}

// rgba texture bound to model materials
// pixels hold straight non premultiplied srgb rgba bytes raw in the color32 container
// so an alpha channel that carries data instead of coverage never darkens the color
#[derive(Clone)]
pub struct Texture2D {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<Color32>,
    // half log2 of the texel count added to the per triangle uv density to pick a mip level
    lod_size: f32,
    mips: OnceLock<Vec<MipLevel>>,
}

#[inline]
fn texel(c: Color32) -> [f32; 4] {
    let [r, g, b, a] = c.to_array();
    [r as f32, g as f32, b as f32, a as f32]
}

// wraps a texel coordinate that is at most one period outside the texture
#[inline]
fn wrap(i: isize, n: usize) -> usize {
    let n = n as isize;
    (if i < 0 { i + n } else if i >= n { i - n } else { i }).clamp(0, n - 1) as usize
}

fn bilinear(w: usize, h: usize, px: &[Color32], u: f32, v: f32, alpha_weighted: bool) -> [f32; 4] {
    let x = (u - u.floor()) * w as f32 - 0.5;
    let y = (v - v.floor()) * h as f32 - 0.5;
    let (xf, yf) = (x.floor(), y.floor());
    let (fx, fy) = (x - xf, y - yf);
    // a nan coordinate saturates to texel zero
    let (x0, y0) = (wrap(xf as isize, w), wrap(yf as isize, h));
    let (x1, y1) = (wrap(x0 as isize + 1, w), wrap(y0 as isize + 1, h));
    let taps = [
        (texel(px[y0 * w + x0]), (1.0 - fx) * (1.0 - fy)),
        (texel(px[y0 * w + x1]), fx * (1.0 - fy)),
        (texel(px[y1 * w + x0]), (1.0 - fx) * fy),
        (texel(px[y1 * w + x1]), fx * fy),
    ];
    let mut plain = [0.0f32; 4];
    for (c, wt) in &taps {
        for k in 0..4 {
            plain[k] += c[k] * wt;
        }
    }
    if alpha_weighted && plain[3] > 0.5 && plain[3] < 254.5 {
        // weight color by coverage so transparent texels do not bleed into the visible edge
        let mut rgb = [0.0f32; 3];
        for (c, wt) in &taps {
            for k in 0..3 {
                rgb[k] += c[k] * wt * c[3];
            }
        }
        let inv = 1.0 / plain[3];
        return [rgb[0] * inv, rgb[1] * inv, rgb[2] * inv, plain[3]];
    }
    plain
}

impl Texture2D {
    pub fn new(w: usize, h: usize, mut pixels: Vec<Color32>) -> Self {
        // sampling indexes by size so a short buffer is padded instead of trusted
        if pixels.len() < w * h {
            pixels.resize(w * h, Color32::WHITE);
        }
        let lod_size = 0.5 * ((w * h).max(1) as f32).log2();
        Self { w, h, pixels, lod_size, mips: OnceLock::new() }
    }

    // true when any texel is not fully opaque
    pub fn has_alpha(&self) -> bool {
        self.pixels.iter().any(|p| p.a() < 255)
    }

    // mip level for a surface that maps uv_per_pixel units of uv area onto one screen sample as half log2
    #[inline]
    pub fn lod_for(&self, half_log2_uv_density: f32) -> f32 {
        half_log2_uv_density + self.lod_size
    }

    // box filtered mip chain below the base level averaged in linear light
    fn mip_chain(&self) -> &[MipLevel] {
        self.mips.get_or_init(|| {
            let to_linear = srgb_to_linear_table();
            let to_srgb = linear_to_srgb_table();
            let mut chain: Vec<MipLevel> = Vec::new();
            let (mut w, mut h) = (self.w, self.h);
            while w > 1 || h > 1 {
                let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
                let src: &[Color32] = chain.last().map_or(&self.pixels, |m| &m.pixels);
                let mut pixels = Vec::with_capacity(nw * nh);
                for y in 0..nh {
                    let (y0, y1) = ((y * 2).min(h - 1), (y * 2 + 1).min(h - 1));
                    for x in 0..nw {
                        let (x0, x1) = ((x * 2).min(w - 1), (x * 2 + 1).min(w - 1));
                        let mut acc = [0.0f32; 4];
                        for i in [y0 * w + x0, y0 * w + x1, y1 * w + x0, y1 * w + x1] {
                            let [r, g, b, a] = src[i].to_array();
                            acc[0] += to_linear[r as usize];
                            acc[1] += to_linear[g as usize];
                            acc[2] += to_linear[b as usize];
                            acc[3] += a as f32;
                        }
                        pixels.push(Color32::from_rgba_premultiplied(
                            encode_srgb(to_srgb, acc[0] * 0.25),
                            encode_srgb(to_srgb, acc[1] * 0.25),
                            encode_srgb(to_srgb, acc[2] * 0.25),
                            (acc[3] * 0.25 + 0.5) as u8,
                        ));
                    }
                }
                chain.push(MipLevel { w: nw, h: nh, pixels });
                (w, h) = (nw, nh);
            }
            chain
        })
    }

    // trilinear filtered straight rgba in 0 to 255 lod 0 is the base level
    // alpha_weighted keeps the color of partly transparent texels from leaking into cutout and blended edges
    pub fn sample_filtered(&self, u: f32, v: f32, lod: f32, alpha_weighted: bool) -> [f32; 4] {
        if self.w == 0 || self.h == 0 {
            return [255.0; 4];
        }
        // a nan lod stays on the base level
        if lod.is_nan() || lod <= 0.01 || (self.w == 1 && self.h == 1) {
            return bilinear(self.w, self.h, &self.pixels, u, v, alpha_weighted);
        }
        let chain = self.mip_chain();
        let lod = lod.min(chain.len() as f32);
        let lo = lod.floor() as usize;
        let frac = lod - lo as f32;
        let level = |i: usize| -> [f32; 4] {
            match i.checked_sub(1).and_then(|m| chain.get(m)) {
                Some(m) => bilinear(m.w, m.h, &m.pixels, u, v, alpha_weighted),
                None => bilinear(self.w, self.h, &self.pixels, u, v, alpha_weighted),
            }
        };
        let a = level(lo);
        if frac < 0.02 || lo >= chain.len() {
            return a;
        }
        let b = level(lo + 1);
        std::array::from_fn(|k| a[k] + (b[k] - a[k]) * frac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(r: u8, g: u8, b: u8, a: u8) -> Color32 {
        Color32::from_rgba_premultiplied(r, g, b, a)
    }

    #[test]
    fn transfer_tables_round_trip_every_srgb_byte() {
        let (lin, srgb) = (srgb_to_linear_table(), linear_to_srgb_table());
        for i in 0..=255u8 {
            assert_eq!(encode_srgb(srgb, lin[i as usize]), i);
        }
        assert_eq!(encode_srgb(srgb, f32::NAN), 0);
        assert_eq!(encode_srgb(srgb, 7.0), 255);
        assert_eq!(encode_srgb(srgb, -1.0), 0);
    }

    #[test]
    fn bilinear_blends_neighbours_and_wraps() {
        let t = Texture2D::new(2, 1, vec![raw(0, 0, 0, 255), raw(200, 100, 50, 255)]);
        // texel centers return the texel itself
        assert_eq!(t.sample_filtered(0.25, 0.5, 0.0, false), [0.0, 0.0, 0.0, 255.0]);
        assert_eq!(t.sample_filtered(0.75, 0.5, 0.0, false), [200.0, 100.0, 50.0, 255.0]);
        // halfway between the centers and across the repeat border
        for u in [0.5, 1.0, -3.0] {
            let c = t.sample_filtered(u, 0.5, 0.0, false);
            assert!((c[0] - 100.0).abs() < 0.01 && (c[1] - 50.0).abs() < 0.01, "u={u} {c:?}");
        }
        // non finite coordinates must not panic
        let _ = t.sample_filtered(f32::NAN, f32::INFINITY, f32::NAN, true);
    }

    #[test]
    fn alpha_as_data_does_not_darken_color_and_cutout_edges_do_not_bleed() {
        // straight storage keeps the color of a texel whose alpha is zero
        let t = Texture2D::new(2, 1, vec![raw(240, 30, 30, 0), raw(240, 30, 30, 255)]);
        let c = t.sample_filtered(0.5, 0.5, 0.0, false);
        assert!((c[0] - 240.0).abs() < 0.01, "opaque use ignores alpha: {c:?}");
        // black transparent neighbour of a punch through texture must not darken the visible edge
        let cut = Texture2D::new(2, 1, vec![raw(0, 0, 0, 0), raw(30, 200, 30, 255)]);
        let plain = cut.sample_filtered(0.5, 0.5, 0.0, false);
        let weighted = cut.sample_filtered(0.5, 0.5, 0.0, true);
        assert!((plain[1] - 100.0).abs() < 0.01 && (weighted[1] - 200.0).abs() < 0.01, "{plain:?} {weighted:?}");
        assert!((weighted[3] - 127.5).abs() < 0.01);
        assert!(cut.has_alpha() && !Texture2D::new(1, 1, vec![raw(1, 2, 3, 255)]).has_alpha());
    }

    #[test]
    fn mips_average_in_linear_light_and_stop_shimmer() {
        // one pixel checkerboard the worst case for minification aliasing
        let n = 64;
        let px: Vec<Color32> = (0..n * n).map(|i| if (i % n + i / n) % 2 == 0 { raw(255, 255, 255, 255) } else { raw(0, 0, 0, 255) }).collect();
        let t = Texture2D::new(n, n, px);
        // texel centers of the base level keep the full checker contrast
        let base = t.sample_filtered(19.5 / 64.0, 49.5 / 64.0, 0.0, false)[0];
        assert!(base == 0.0 || base == 255.0, "base level texel center: {base}");
        // half white half black is 0.5 in linear light which is srgb 188 not 128
        for lod in [1.0, 2.0, 3.5, 6.0, 40.0] {
            let c = t.sample_filtered(0.31, 0.77, lod, false);
            assert!((c[0] - 188.0).abs() < 1.5 && c[3] == 255.0, "lod {lod} gave {c:?}");
        }
        assert_eq!(t.mip_chain().len(), 6);
        assert_eq!(t.mip_chain().last().map(|m| (m.w, m.h)), Some((1, 1)));
        // non square and odd sizes reduce down to a single texel
        let odd = Texture2D::new(5, 3, vec![raw(10, 20, 30, 255); 15]);
        let c = odd.sample_filtered(0.5, 0.5, 9.0, false);
        assert!((c[0] - 10.0).abs() <= 1.0 && (c[2] - 30.0).abs() <= 1.0);
        // a short pixel buffer is padded rather than indexed out of bounds
        let short = Texture2D::new(4, 4, vec![raw(9, 9, 9, 255); 3]);
        assert_eq!(short.pixels.len(), 16);
        let _ = short.sample_filtered(0.9, 0.9, 1.0, false);
    }
}
