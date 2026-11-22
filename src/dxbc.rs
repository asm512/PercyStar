use flate2::read::ZlibDecoder;
use std::ffi::c_void;
use std::fmt::Write as _;
use std::io::Read;
#[cfg(windows)]
use std::sync::OnceLock;

const DXBC_MAGIC: &[u8; 4] = b"DXBC";
const FSSM_MAGIC: &[u8; 4] = b"FSSM";
const GFX103_CODE_START: [u8; 4] = 0xBFA0_0003u32.to_le_bytes();
const GFX103_CODE_END: [u8; 4] = 0xBF81_0000u32.to_le_bytes();

#[cfg(windows)]
const CLSID_DXC_COMPILER: Guid = Guid::from_values(
    0x73e22d93,
    0xe6ce,
    0x47f3,
    [0xb5, 0xbf, 0xf0, 0x66, 0x4f, 0x39, 0xc1, 0xb0],
);

#[cfg(windows)]
const IID_IDXC_COMPILER3: Guid = Guid::from_values(
    0x228b4687,
    0x5a6a,
    0x4730,
    [0x90, 0x0c, 0x97, 0x02, 0xb2, 0x20, 0x3f, 0x54],
);

#[cfg(windows)]
const IID_IDXC_RESULT: Guid = Guid::from_values(
    0x58346cda,
    0xdde7,
    0x4497,
    [0x94, 0x61, 0x6f, 0x87, 0xaf, 0x5e, 0x06, 0x59],
);

#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

#[cfg(windows)]
impl Guid {
    const fn from_values(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Self {
        Self {
            data1,
            data2,
            data3,
            data4,
        }
    }
}

#[cfg(windows)]
#[repr(C)]
struct DxcBuffer {
    ptr: *const c_void,
    size: usize,
    encoding: u32,
}

pub fn disassemble(data: &[u8]) -> String {
    if let Ok(fssm) = extract_fssm(data) {
        return format_fssm_disasm(&fssm, "DXBC / FSSM");
    }

    match disassemble_with_dxc(data) {
        Ok(text) => text,
        Err(error) => format!("{error}\n\n{}", inspect(data)),
    }
}

#[cfg(test)]
pub fn disassemble_with_source(data: &[u8], source: &str) -> String {
    match extract_fssm(data) {
        Ok(fssm) => format_fssm_disasm(&fssm, source),
        Err(_) => disassemble(data),
    }
}

#[derive(Clone, Debug)]
pub struct FssmShader {
    pub data: Vec<u8>,
    pub code_offset: usize,
    pub code_size: usize,
}

impl FssmShader {
    pub fn code(&self) -> &[u8] {
        let end = (self.code_offset + self.code_size).min(self.data.len());
        &self.data[self.code_offset..end]
    }
}

pub fn extract_fssm(data: &[u8]) -> Result<FssmShader, String> {
    let payload = find_dxil_payload(data)?;

    // probe common nobc zlib offsets first before scanning 0x60..0xc0
    let scan_end = payload.len().min(0xC0).saturating_sub(2);
    for offset in [0x64usize, 0xA0].into_iter().chain(0x60..=scan_end) {
        if offset > scan_end || !is_zlib_header(&payload[offset..offset + 2]) {
            continue;
        }

        // bounded inflate a shader never comes close to this
        let mut decompressed = Vec::new();
        if ZlibDecoder::new(&payload[offset..])
            .take(64 << 20)
            .read_to_end(&mut decompressed)
            .is_err()
            || decompressed.len() < 4
            || &decompressed[..4] != FSSM_MAGIC
        {
            continue;
        }

        let code_offset = find_bytes(&decompressed, &GFX103_CODE_START)
            .ok_or_else(|| "FSSM has no GFX10.3 code start marker".to_string())?;
        let code_size = decompressed[code_offset + 4..]
            .windows(4)
            .position(|window| window == GFX103_CODE_END)
            .map(|end_offset| end_offset + 8)
            .unwrap_or_else(|| decompressed.len() - code_offset);

        return Ok(FssmShader {
            data: decompressed,
            code_offset,
            code_size,
        });
    }

    Err("DXIL payload does not contain a zlib FSSM stream".into())
}

#[cfg(windows)]
fn disassemble_with_dxc(data: &[u8]) -> Result<String, String> {
    if data.len() < 32 || &data[..4] != DXBC_MAGIC {
        return Err("not a DXBC container".into());
    }

    let create_instance = cached_dxc_create_instance()?;
    let mut compiler = std::ptr::null_mut::<c_void>();
    unsafe {
        let status = create_instance(
            &CLSID_DXC_COMPILER as *const Guid as *const c_void,
            &IID_IDXC_COMPILER3 as *const Guid as *const c_void,
            &mut compiler,
        );
        if status != 0 {
            return Err(format!("DxcCreateInstance failed: 0x{status:08x}"));
        }
    }

    let input = DxcBuffer {
        ptr: data.as_ptr() as *const c_void,
        size: data.len(),
        encoding: 0,
    };
    let mut result = std::ptr::null_mut::<c_void>();
    unsafe {
        let disassemble = com_call::<
            unsafe extern "system" fn(
                *mut c_void,
                *const DxcBuffer,
                *const Guid,
                *mut *mut c_void,
            ) -> i32,
        >(compiler, 4);
        let status = disassemble(compiler, &input, &IID_IDXC_RESULT, &mut result);
        release(compiler);
        if status != 0 {
            return Err(format!("IDxcCompiler3::Disassemble failed: 0x{status:08x}"));
        }
    }

    let mut operation_status = 0i32;
    let mut blob = std::ptr::null_mut::<c_void>();
    unsafe {
        com_call::<unsafe extern "system" fn(*mut c_void, *mut i32) -> i32>(result, 3)(
            result,
            &mut operation_status,
        );
        com_call::<unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32>(result, 4)(
            result, &mut blob,
        );
        release(result);
    }

    if operation_status != 0 {
        if !blob.is_null() {
            unsafe { release(blob) };
        }
        return Err(format!("DXC disassembly failed: 0x{operation_status:08x}"));
    }
    if blob.is_null() {
        return Err("DXC returned no disassembly blob".into());
    }

    unsafe {
        let get_pointer =
            com_call::<unsafe extern "system" fn(*mut c_void) -> *mut c_void>(blob, 3);
        let get_size = com_call::<unsafe extern "system" fn(*mut c_void) -> usize>(blob, 4);
        let pointer = get_pointer(blob);
        let size = get_size(blob);
        let bytes = std::slice::from_raw_parts(pointer as *const u8, size);
        let text = String::from_utf8_lossy(bytes).into_owned();
        release(blob);
        Ok(text)
    }
}

#[cfg(not(windows))]
fn disassemble_with_dxc(_data: &[u8]) -> Result<String, String> {
    Err("DXC disassembly is only available on Windows with dxcompiler.dll".into())
}

#[cfg(windows)]
unsafe fn com_call<F>(instance: *mut c_void, slot: usize) -> F {
    debug_assert!(!instance.is_null());
    let vtable = *(instance as *const *const *mut c_void);
    let function = *vtable.add(slot);
    std::mem::transmute_copy(&function)
}

#[cfg(windows)]
unsafe fn release(instance: *mut c_void) {
    if instance.is_null() {
        return;
    }
    let release = com_call::<unsafe extern "system" fn(*mut c_void) -> u32>(instance, 2);
    release(instance);
}

#[cfg(windows)]
type DxcCreateInstanceFn =
    unsafe extern "system" fn(*const c_void, *const c_void, *mut *mut c_void) -> i32;

#[cfg(windows)]
fn cached_dxc_create_instance() -> Result<DxcCreateInstanceFn, String> {
    static DXC_CREATE: OnceLock<Result<DxcCreateInstanceFn, String>> = OnceLock::new();
    DXC_CREATE
        .get_or_init(|| {
            let library = load_dxc_library()?;
            get_dxc_create_instance(library)
        })
        .clone()
}

#[cfg(windows)]
fn load_dxc_library() -> Result<*mut c_void, String> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    }
    // a bare name would also search the working directory and path where a dump folder can plant a dll
    const SEARCH_DLL_LOAD_DIR: u32 = 0x100;
    const SEARCH_APPLICATION_DIR: u32 = 0x200;
    const SEARCH_SYSTEM32: u32 = 0x800;
    let load = |name: &std::ffi::OsStr, flags: u32| {
        let wide: Vec<u16> = name.encode_wide().chain(std::iter::once(0)).collect();
        unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), flags) }
    };
    // explicit full path or the folder of the executable and system32
    let explicit = std::env::var_os("DXCOMPILER_DLL").map(std::path::PathBuf::from).filter(|p| p.is_absolute() && p.is_file());
    let library = match explicit {
        Some(path) => load(path.as_os_str(), SEARCH_DLL_LOAD_DIR | SEARCH_SYSTEM32),
        None => load(std::ffi::OsStr::new("dxcompiler.dll"), SEARCH_APPLICATION_DIR | SEARCH_SYSTEM32),
    };
    if library.is_null() {
        return Err("dxcompiler.dll was not found next to PercyStar.exe (or set DXCOMPILER_DLL to its full path)".into());
    }
    Ok(library)
}

#[cfg(windows)]
fn get_dxc_create_instance(
    library: *mut c_void,
) -> Result<unsafe extern "system" fn(*const c_void, *const c_void, *mut *mut c_void) -> i32, String>
{
    #[link(name = "kernel32")]
    extern "system" {
        fn GetProcAddress(library: isize, name: *const u8) -> *const c_void;
    }

    let name = c"DxcCreateInstance";
    let address = unsafe { GetProcAddress(library as isize, name.as_ptr() as *const u8) };
    if address.is_null() {
        return Err("dxcompiler.dll does not export DxcCreateInstance".into());
    }
    Ok(unsafe { std::mem::transmute(address) })
}

fn find_dxil_payload(data: &[u8]) -> Result<&[u8], String> {
    if data.len() < 0x24 || &data[..4] != DXBC_MAGIC {
        return Err("not a DXBC container".into());
    }

    let read_u32 = |offset: usize| {
        u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    };
    let chunk_count = read_u32(0x1c) as usize;

    for chunk_index in 0..chunk_count {
        let table_offset = 0x20 + chunk_index * 4;
        if table_offset + 4 > data.len() {
            break;
        }
        let chunk_offset = read_u32(table_offset) as usize;
        if chunk_offset + 8 > data.len() || &data[chunk_offset..chunk_offset + 4] != b"DXIL" {
            continue;
        }

        let size = read_u32(chunk_offset + 4) as usize;
        let start = chunk_offset + 8;
        let end = start.checked_add(size).ok_or("invalid DXIL chunk size")?;
        if end <= data.len() {
            return Ok(&data[start..end]);
        }
    }

    Err("DXBC container has no valid DXIL chunk".into())
}

fn is_zlib_header(header: &[u8]) -> bool {
    header.len() == 2
        && header[0] & 0x0f == 8
        && header[0] >> 4 <= 7
        && header[1] & 0x20 == 0
        && (u16::from(header[0]) << 8 | u16::from(header[1])) % 31 == 0
}

fn find_bytes(data: &[u8], needle: &[u8]) -> Option<usize> {
    data.windows(needle.len())
        .position(|window| window == needle)
}

fn format_fssm_disasm(fssm: &FssmShader, source: &str) -> String {
    let instructions = crate::rdna2::decode_gfx103(fssm.code(), 4000);
    let mut output = String::with_capacity(instructions.len() * 48 + 512);
    let _ = writeln!(
        output,
        "; =============================================================================\n\
         ; PercyStar AMD RDNA2 (GFX10.3) Disassembly\n\
         ; Source: {source}\n\
         ; FSSM: {} bytes | Code: 0x{:X} + {} bytes | {} insns\n\
         ; =============================================================================\n",
        fssm.data.len(),
        fssm.code_offset,
        fssm.code_size,
        instructions.len()
    );

    for instruction in &instructions {
        let mut words = String::new();
        for word in 0..instruction.nwords {
            let _ = write!(words, "{:08x} ", instruction.words[word]);
        }
        let _ = writeln!(
            output,
            "  +{:04x}: {:<28} {:<26} {}",
            instruction.offset, words, instruction.mnemonic, instruction.operands
        );
    }
    output
}

pub fn inspect(data: &[u8]) -> String {
    if data.len() < 0x24 || &data[..4] != DXBC_MAGIC {
        return "Not a DXBC container".into();
    }

    let read_u32 = |offset: usize| {
        u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    };
    let chunk_count = read_u32(0x1c) as usize;
    let mut output = String::new();
    let _ = writeln!(
        output,
        "DXBC container: {} bytes, {} chunks",
        data.len(),
        chunk_count
    );
    for chunk_index in 0..chunk_count {
        let table_offset = 0x20 + chunk_index * 4;
        if table_offset + 4 > data.len() {
            let _ = writeln!(output, "  truncated chunk table at 0x{table_offset:x}");
            break;
        }
        let offset = read_u32(table_offset) as usize;
        if offset + 8 > data.len() {
            let _ = writeln!(output, "  invalid chunk offset 0x{offset:x}");
            continue;
        }
        let tag = &data[offset..offset + 4];
        let size = read_u32(offset + 4) as usize;
        let end = offset + 8 + size;
        let state = if end <= data.len() { "ok" } else { "truncated" };
        let _ = writeln!(
            output,
            "  {} at 0x{:08x}, {} bytes ({state})",
            String::from_utf8_lossy(tag),
            offset,
            size
        );
        if tag == b"DXIL" && end <= data.len() {
            let content = &data[offset + 8..end];
            if content.windows(4).any(|window| window == b"NOBC") {
                let _ = writeln!(
                    output,
                    "    NOBC wrapped shader: DXIL payload uses NOBC (no LLVM bitcode)"
                );
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write as _;

    #[test]
    fn inspects_dxbc_chunk_table() {
        let mut data = vec![0u8; 0x24];
        data[..4].copy_from_slice(DXBC_MAGIC);
        data[0x1c..0x20].copy_from_slice(&1u32.to_le_bytes());
        data[0x20..0x24].copy_from_slice(&0x24u32.to_le_bytes());
        data.extend_from_slice(b"TEST");
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&[1, 2, 3, 4]);

        let text = inspect(&data);
        assert!(text.contains("1 chunks"));
        assert!(text.contains("TEST"));
    }

    #[test]
    fn extracts_and_disassembles_fssm_from_dxil() {
        let mut fssm = vec![0u8; 0x84];
        fssm[..4].copy_from_slice(FSSM_MAGIC);
        fssm.extend_from_slice(&GFX103_CODE_START);
        fssm.extend_from_slice(&GFX103_CODE_END);

        let mut payload = vec![0u8; 0x7c];
        payload[..4].copy_from_slice(b"XBCH");
        let mut compressor = ZlibEncoder::new(Vec::new(), Compression::fast());
        compressor.write_all(&fssm).unwrap();
        payload.extend_from_slice(&compressor.finish().unwrap());

        let mut data = vec![0u8; 0x24];
        data[..4].copy_from_slice(DXBC_MAGIC);
        data[0x1c..0x20].copy_from_slice(&1u32.to_le_bytes());
        data[0x20..0x24].copy_from_slice(&0x24u32.to_le_bytes());
        data.extend_from_slice(b"DXIL");
        data.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        data.extend_from_slice(&payload);

        let extracted = extract_fssm(&data).unwrap();
        assert_eq!(extracted.data, fssm);
        assert_eq!(extracted.code_offset, 0x84);
        assert_eq!(extracted.code_size, 8);

        let text = disassemble_with_source(&data, "synthetic SGD5 shader");
        assert!(text.contains("s_setprio"));
        assert!(text.contains("s_endpgm"));
        assert!(text.contains("synthetic SGD5 shader"));
    }
}
