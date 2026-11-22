//! Zero-dependency pure-Rust parser for Bink (`BIK*`) and Bink2 (`KB2*`)
//! video container headers, audio track tables, and frame offset tables.

/// Maximum number of audio tracks accepted when parsing a container header.
pub const MAX_AUDIO_TRACKS: u32 = 64;

/// Size of the fixed Bink / Bink2 container header in bytes (11 x 32-bit words).
const HEADER_SIZE: usize = 44;

/// Parsed Bink / Bink2 container header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinkContainerHeader {
    pub magic: [u8; 4],
    pub revision: char,
    pub file_size: u32,
    pub frame_count: u32,
    pub largest_frame_size: u32,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_div: u32,
    pub video_flags: u32,
    pub audio_track_count: u32,
}

impl BinkContainerHeader {
    /// Returns `true` if the container uses Bink2 (`KB2*`) magic.
    pub fn is_bink2(&self) -> bool {
        &self.magic[..3] == b"KB2"
    }

    /// Returns `true` if the container uses Bink1 (`BIK*`) magic.
    pub fn is_bink1(&self) -> bool {
        &self.magic[..3] == b"BIK"
    }

    /// Returns the frame rate in frames per second, or `0.0` if `fps_div == 0`.
    pub fn fps(&self) -> f64 {
        if self.fps_div == 0 {
            0.0
        } else {
            self.fps_num as f64 / self.fps_div as f64
        }
    }
}

/// Audio track metadata entry from the Bink container audio table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinkAudioTrack {
    pub max_size: u32,
    pub sample_rate: u16,
    pub flags: u16,
    pub track_id: u32,
}

/// Single frame entry decoded from the Bink frame offset table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinkFrameEntry {
    pub index: usize,
    pub offset: u32,
    pub size: u32,
    pub is_keyframe: bool,
}

/// Complete parsed Bink / Bink2 container metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BinkContainer {
    pub header: BinkContainerHeader,
    pub audio_tracks: Vec<BinkAudioTrack>,
    pub frames: Vec<BinkFrameEntry>,
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let end = self
            .pos
            .checked_add(N)
            .ok_or_else(|| "Buffer offset overflow".to_string())?;
        if end > self.data.len() {
            return Err(format!(
                "Unexpected EOF at offset {}: need {} bytes, have {}",
                self.pos,
                N,
                self.remaining()
            ));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.data[self.pos..end]);
        self.pos = end;
        Ok(out)
    }

    fn read_u16_le(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.read_bytes::<2>()?))
    }

    fn read_u32_le(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.read_bytes::<4>()?))
    }
}

impl BinkContainer {
    /// Parses a Bink (`BIK*`) or Bink2 (`KB2*`) container from raw bytes.
    pub fn parse(data: &[u8]) -> Result<Self, String> {
        if data.len() < HEADER_SIZE {
            return Err(format!(
                "Input too short for Bink header: {} bytes (need at least {})",
                data.len(),
                HEADER_SIZE
            ));
        }

        let mut cur = Cursor::new(data);
        let magic = cur.read_bytes::<4>()?;
        if &magic[..3] != b"KB2" && &magic[..3] != b"BIK" {
            return Err(format!(
                "Invalid Bink magic {:?}; expected KB2* or BIK*",
                String::from_utf8_lossy(&magic)
            ));
        }
        let revision = magic[3] as char;

        let file_size = cur.read_u32_le()?;
        let frame_count = cur.read_u32_le()?;
        let largest_frame_size = cur.read_u32_le()?;
        let _internal_frame_count = cur.read_u32_le()?;
        let width = cur.read_u32_le()?;
        let height = cur.read_u32_le()?;
        let fps_num = cur.read_u32_le()?;
        let fps_div = cur.read_u32_le()?;
        let video_flags = cur.read_u32_le()?;
        let audio_track_count = cur.read_u32_le()?;

        if width == 0 || height == 0 {
            return Err(format!("Invalid video dimensions: {}x{}", width, height));
        }
        if audio_track_count > MAX_AUDIO_TRACKS {
            return Err(format!(
                "Audio track count {} exceeds maximum {}",
                audio_track_count, MAX_AUDIO_TRACKS
            ));
        }

        let track_count = audio_track_count as usize;
        let audio_table_bytes = track_count
            .checked_mul(12)
            .ok_or_else(|| "Audio track table size overflow".to_string())?;
        if cur.remaining() < audio_table_bytes {
            return Err(format!(
                "Truncated audio track table: need {} bytes for {} tracks, have {}",
                audio_table_bytes,
                track_count,
                cur.remaining()
            ));
        }

        let mut audio_tracks = Vec::with_capacity(track_count);
        if track_count > 0 {
            let mut max_sizes = Vec::with_capacity(track_count);
            for _ in 0..track_count {
                max_sizes.push(cur.read_u32_le()?);
            }

            let mut rates_and_flags = Vec::with_capacity(track_count);
            for _ in 0..track_count {
                let sample_rate = cur.read_u16_le()?;
                let flags = cur.read_u16_le()?;
                rates_and_flags.push((sample_rate, flags));
            }

            for i in 0..track_count {
                let track_id = cur.read_u32_le()?;
                audio_tracks.push(BinkAudioTrack {
                    max_size: max_sizes[i],
                    sample_rate: rates_and_flags[i].0,
                    flags: rates_and_flags[i].1,
                    track_id,
                });
            }
        }

        let frame_count_usize = usize::try_from(frame_count)
            .map_err(|_| "Frame count overflows usize".to_string())?;
        let offset_entries = frame_count_usize
            .checked_add(1)
            .ok_or_else(|| "Frame offset table entry count overflow".to_string())?;
        let frame_table_bytes = offset_entries
            .checked_mul(4)
            .ok_or_else(|| "Frame offset table size overflow".to_string())?;
        if cur.remaining() < frame_table_bytes {
            return Err(format!(
                "Truncated frame offset table: need {} bytes for {} entries, have {}",
                frame_table_bytes,
                offset_entries,
                cur.remaining()
            ));
        }

        let mut raw_offsets = Vec::with_capacity(offset_entries);
        for _ in 0..offset_entries {
            raw_offsets.push(cur.read_u32_le()?);
        }

        let mut frames = Vec::with_capacity(frame_count_usize);
        for i in 0..frame_count_usize {
            let raw = raw_offsets[i];
            let is_keyframe = (raw & 1) != 0;
            let offset = raw & !1;
            let next_offset = raw_offsets[i + 1] & !1;
            let size = next_offset.checked_sub(offset).ok_or_else(|| {
                format!(
                    "Non-monotonic frame offset at frame {}: {} -> {}",
                    i, offset, next_offset
                )
            })?;
            frames.push(BinkFrameEntry {
                index: i,
                offset,
                size,
                is_keyframe,
            });
        }

        Ok(Self {
            header: BinkContainerHeader {
                magic,
                revision,
                file_size,
                frame_count,
                largest_frame_size,
                width,
                height,
                fps_num,
                fps_div,
                video_flags,
                audio_track_count,
            },
            audio_tracks,
            frames,
        })
    }

    /// Formats a human-readable inspection report of the container header,
    /// audio tracks, and frame table summary.
    pub fn format_report(&self) -> String {
        let h = &self.header;
        let magic_str = String::from_utf8_lossy(&h.magic);
        let codec = if h.is_bink2() {
            "Bink2"
        } else if h.is_bink1() {
            "Bink1"
        } else {
            "Unknown"
        };
        let keyframe_count = self.frames.iter().filter(|f| f.is_keyframe).count();

        let mut out = String::new();
        out.push_str(&format!(
            "=== {} Container ({}, rev '{}') ===\n",
            codec, magic_str, h.revision
        ));
        out.push_str(&format!(
            "Resolution: {}x{} | FPS: {}/{} ({:.3} fps)\n",
            h.width,
            h.height,
            h.fps_num,
            h.fps_div,
            h.fps()
        ));
        out.push_str(&format!(
            "Frames: {} (keyframes: {}) | Largest frame: {} bytes | File size field: {}\n",
            h.frame_count, keyframe_count, h.largest_frame_size, h.file_size
        ));
        out.push_str(&format!(
            "Video flags: 0x{:08X} | Audio tracks: {}\n",
            h.video_flags, h.audio_track_count
        ));

        for (i, track) in self.audio_tracks.iter().enumerate() {
            out.push_str(&format!(
                "  Audio[{}]: id={}, sample_rate={} Hz, flags=0x{:04X}, max_size={}\n",
                i, track.track_id, track.sample_rate, track.flags, track.max_size
            ));
        }

        let preview = self.frames.len().min(8);
        if preview > 0 {
            out.push_str(&format!("Frame table (showing {}/{}):\n", preview, self.frames.len()));
            for f in &self.frames[..preview] {
                out.push_str(&format!(
                    "  Frame[{:04}]: offset=0x{:08X}, size={} bytes, keyframe={}\n",
                    f.index, f.offset, f.size, f.is_keyframe
                ));
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn push_u16(buf: &mut Vec<u8>, v: u16) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn build_synthetic_bink(
        magic: [u8; 4],
        width: u32,
        height: u32,
        fps_num: u32,
        fps_div: u32,
        audio_tracks: &[(u32, u16, u16, u32)],
        raw_frame_offsets: &[u32],
    ) -> Vec<u8> {
        let frame_count = raw_frame_offsets.len().saturating_sub(1) as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&magic);
        push_u32(&mut buf, 4096); // file_size
        push_u32(&mut buf, frame_count);
        push_u32(&mut buf, 1024); // largest_frame_size
        push_u32(&mut buf, frame_count); // internal frame count
        push_u32(&mut buf, width);
        push_u32(&mut buf, height);
        push_u32(&mut buf, fps_num);
        push_u32(&mut buf, fps_div);
        push_u32(&mut buf, 0x0000_0001); // video_flags
        push_u32(&mut buf, audio_tracks.len() as u32);

        for (max_size, _, _, _) in audio_tracks {
            push_u32(&mut buf, *max_size);
        }
        for (_, sample_rate, flags, _) in audio_tracks {
            push_u16(&mut buf, *sample_rate);
            push_u16(&mut buf, *flags);
        }
        for (_, _, _, track_id) in audio_tracks {
            push_u32(&mut buf, *track_id);
        }
        for &off in raw_frame_offsets {
            push_u32(&mut buf, off);
        }
        buf
    }

    #[test]
    fn parses_kb2n_with_audio_and_keyframes() {
        let data = build_synthetic_bink(
            *b"KB2n",
            1920,
            1080,
            60,
            1,
            &[(8192, 48000, 0x0020, 100), (4096, 44100, 0x0000, 101)],
            &[
                100 | 1, // frame 0: offset 100, keyframe
                350,     // frame 1: offset 350, inter
                600 | 1, // frame 2: offset 600, keyframe
                900,     // sentinel end offset 900
            ],
        );

        let parsed = BinkContainer::parse(&data).expect("valid KB2n container");
        assert!(parsed.header.is_bink2());
        assert!(!parsed.header.is_bink1());
        assert_eq!(parsed.header.revision, 'n');
        assert_eq!(parsed.header.width, 1920);
        assert_eq!(parsed.header.height, 1080);
        assert_eq!(parsed.header.frame_count, 3);
        assert_eq!(parsed.header.audio_track_count, 2);

        assert_eq!(parsed.audio_tracks.len(), 2);
        assert_eq!(
            parsed.audio_tracks[0],
            BinkAudioTrack {
                max_size: 8192,
                sample_rate: 48000,
                flags: 0x0020,
                track_id: 100,
            }
        );
        assert_eq!(
            parsed.audio_tracks[1],
            BinkAudioTrack {
                max_size: 4096,
                sample_rate: 44100,
                flags: 0x0000,
                track_id: 101,
            }
        );

        assert_eq!(parsed.frames.len(), 3);
        assert_eq!(
            parsed.frames[0],
            BinkFrameEntry {
                index: 0,
                offset: 100,
                size: 250,
                is_keyframe: true,
            }
        );
        assert_eq!(
            parsed.frames[1],
            BinkFrameEntry {
                index: 1,
                offset: 350,
                size: 250,
                is_keyframe: false,
            }
        );
        assert_eq!(
            parsed.frames[2],
            BinkFrameEntry {
                index: 2,
                offset: 600,
                size: 300,
                is_keyframe: true,
            }
        );

        let report = parsed.format_report();
        assert!(report.contains("Bink2 Container (KB2n, rev 'n')"));
        assert!(report.contains("1920x1080"));
        assert!(report.contains("Audio[0]: id=100, sample_rate=48000 Hz"));
    }

    #[test]
    fn parses_biki_without_audio() {
        let data = build_synthetic_bink(*b"BIKi", 1280, 720, 30000, 1001, &[], &[64 | 1, 200, 320]);
        let parsed = BinkContainer::parse(&data).expect("valid BIKi container");
        assert!(parsed.header.is_bink1());
        assert_eq!(parsed.header.revision, 'i');
        assert!(parsed.audio_tracks.is_empty());
        assert_eq!(parsed.frames.len(), 2);
        assert_eq!(parsed.frames[0].offset, 64);
        assert_eq!(parsed.frames[0].size, 136);
        assert!(parsed.frames[0].is_keyframe);
        assert_eq!(parsed.frames[1].size, 120);
        assert!(!parsed.frames[1].is_keyframe);
    }

    #[test]
    fn rejects_invalid_magic_and_truncation() {
        let short = [0u8; 20];
        assert!(BinkContainer::parse(&short).is_err());

        let mut bad_magic = build_synthetic_bink(*b"KB2n", 640, 480, 30, 1, &[], &[64, 128]);
        bad_magic[0..4].copy_from_slice(b"RIFF");
        assert!(BinkContainer::parse(&bad_magic).is_err());

        let valid = build_synthetic_bink(
            *b"KB2n",
            640,
            480,
            30,
            1,
            &[(1024, 48000, 0, 1)],
            &[100 | 1, 200],
        );
        // Truncate inside audio table
        assert!(BinkContainer::parse(&valid[..HEADER_SIZE + 4]).is_err());
        // Truncate inside frame offset table
        assert!(BinkContainer::parse(&valid[..valid.len() - 4]).is_err());
    }

    #[test]
    fn rejects_excessive_audio_tracks_and_non_monotonic_offsets() {
        let mut too_many_audio = build_synthetic_bink(*b"KB2n", 640, 480, 30, 1, &[], &[64, 128]);
        too_many_audio[40..44].copy_from_slice(&(MAX_AUDIO_TRACKS + 1).to_le_bytes());
        assert!(BinkContainer::parse(&too_many_audio).is_err());

        let non_monotonic = build_synthetic_bink(*b"BIKi", 640, 480, 30, 1, &[], &[300 | 1, 200]);
        assert!(BinkContainer::parse(&non_monotonic).is_err());
    }
}
