#[cfg(windows)]
use std::ffi::CString;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinkInfo {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps_num: u32,
    pub fps_div: u32,
}

#[cfg(any(windows, test))]
pub fn validate_bink_prefix(words: &[u32; 7]) -> Option<BinkInfo> {
    let [w, h, frames, _cur, _last, fps_num, fps_div] = *words;
    if (1..=16384).contains(&w) && (1..=16384).contains(&h) && frames > 0 {
        Some(BinkInfo { width: w, height: h, frames, fps_num, fps_div: fps_div.max(1) })
    } else {
        None
    }
}

#[cfg(windows)]
mod imp {
    use super::*;
    use std::ffi::c_void;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(lpLibFileName: *const u16) -> isize;
        fn GetProcAddress(hModule: isize, lpProcName: *const u8) -> *const c_void;
        fn FreeLibrary(hLibModule: isize) -> i32;
    }

    type BinkOpenFn = unsafe extern "system" fn(*const i8, u32) -> *mut c_void;
    type BinkCloseFn = unsafe extern "system" fn(*mut c_void);
    type BinkDoFrameFn = unsafe extern "system" fn(*mut c_void);
    type BinkNextFrameFn = unsafe extern "system" fn(*mut c_void);
    type BinkCopyToBufferFn = unsafe extern "system" fn(*mut c_void, *mut u8, i32, u32, u32, u32, u32) -> i32;

    const BINKSURFACE32: u32 = 3;

    pub fn decode_bk2_to_pngs(dll_path: &Path, bk2_path: &Path, out_dir: &Path, max_frames: usize) -> Result<(BinkInfo, usize), String> {
        let mut wide: Vec<u16> = dll_path.as_os_str().to_string_lossy().encode_utf16().collect();
        wide.push(0);
        let hmod = unsafe { LoadLibraryW(wide.as_ptr()) };
        if hmod == 0 {
            return Err(format!("Failed to load {}", dll_path.display()));
        }

        let res = (|| {
            let get_fn = |name: &[u8]| -> Result<*const c_void, String> {
                let p = unsafe { GetProcAddress(hmod, name.as_ptr()) };
                if p.is_null() {
                    Err(format!("Missing export {}", String::from_utf8_lossy(name)))
                } else {
                    Ok(p)
                }
            };
            let bink_open: BinkOpenFn = unsafe { std::mem::transmute(get_fn(b"BinkOpen\0")?) };
            let bink_close: BinkCloseFn = unsafe { std::mem::transmute(get_fn(b"BinkClose\0")?) };
            let bink_do: BinkDoFrameFn = unsafe { std::mem::transmute(get_fn(b"BinkDoFrame\0")?) };
            let bink_next: BinkNextFrameFn = unsafe { std::mem::transmute(get_fn(b"BinkNextFrame\0")?) };
            let bink_copy: BinkCopyToBufferFn = unsafe { std::mem::transmute(get_fn(b"BinkCopyToBuffer\0")?) };

            let c_bk2 = CString::new(bk2_path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
            let hbink = unsafe { bink_open(c_bk2.as_ptr(), 0) };
            if hbink.is_null() {
                return Err(format!("BinkOpen failed for {}", bk2_path.display()));
            }
            let words = unsafe { *(hbink as *const [u32; 7]) };
            let info = validate_bink_prefix(&words).ok_or_else(|| {
                unsafe { bink_close(hbink) };
                "Invalid BINK struct header".to_string()
            })?;

            std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
            let stride = (info.width * 4) as usize;
            let mut buf = vec![0u8; stride * info.height as usize];
            let count = (info.frames as usize).min(max_frames.max(1));
            for idx in 0..count {
                unsafe {
                    bink_do(hbink);
                    bink_copy(hbink, buf.as_mut_ptr(), stride as i32, info.height, 0, 0, BINKSURFACE32);
                }
                for px in buf.as_chunks_mut::<4>().0 {
                    px.swap(0, 2); // BGRx -> RGBx
                    px[3] = 255;
                }
                let out_png = out_dir.join(format!("frame_{idx:05}.png"));
                image::save_buffer(&out_png, &buf, info.width, info.height, image::ColorType::Rgba8)
                    .map_err(|e| e.to_string())?;
                if idx + 1 < count {
                    unsafe { bink_next(hbink) };
                }
            }
            unsafe { bink_close(hbink) };
            Ok((info, count))
        })();

        unsafe { FreeLibrary(hmod) };
        res
    }
}

#[cfg(not(windows))]
mod imp {
    use super::*;
    pub fn decode_bk2_to_pngs(_: &Path, _: &Path, _: &Path, _: usize) -> Result<(BinkInfo, usize), String> {
        Err("bink2w64.dll decoding is Windows-only".into())
    }
}

pub use imp::decode_bk2_to_pngs;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_bink_prefix_header() {
        let ok = validate_bink_prefix(&[1920, 1080, 120, 1, 0, 30, 1]).unwrap();
        assert_eq!(ok.width, 1920);
        assert_eq!(ok.height, 1080);
        assert_eq!(ok.frames, 120);
        assert!(validate_bink_prefix(&[0, 1080, 10, 1, 0, 30, 1]).is_none());
    }
}
