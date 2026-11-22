"""decode adat v1 awc audio into playable wav streams"""
import glob
import os
import struct
import subprocess
import sys


def crc32_ogg(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if (crc & 0x80000000) else (crc << 1) & 0xFFFFFFFF
    return crc


def ogg_page(serial: int, seq: int, header_type: int, granule: int, packets: list[bytes]) -> bytes:
    seg_table = bytearray()
    body = bytearray()
    for p in packets:
        rem = len(p)
        while rem >= 255:
            seg_table.append(255)
            rem -= 255
        seg_table.append(rem)
        body.extend(p)
    hdr = struct.pack(
        "<4sBBqIIIB",
        b"OggS", 0, header_type, granule, serial, seq, 0, len(seg_table),
    ) + seg_table
    full = bytearray(hdr + body)
    crc = crc32_ogg(full)
    struct.pack_into("<I", full, 22, crc)
    return bytes(full)


def ru32(b: bytes, o: int) -> int:
    return struct.unpack_from("<I", b, o)[0]


def probe_superblock_streams(blob: bytes):
    """checks if a 0x55 data chunk starts with a 512kb superblock directory
    (0xffffffff marker + u32 block count, per substream, 24 bytes each) --
    ported from src/awc.rs::probe_superblock_streams so the python helper
    stays in sync with the native decoder instead of silently drifting."""
    if len(blob) < 4096:
        return None
    n = 0
    total_blks = 0
    limit = min(len(blob), 2048)
    while (n + 1) * 24 <= limit:
        if ru32(blob, n * 24) != 0xFFFFFFFF:
            break
        cnt = ru32(blob, n * 24 + 4)
        if cnt == 0 or cnt > 255:
            break
        total_blks += cnt
        n += 1
    return n if (n >= 1 and total_blks <= 255) else None


def collect_2kb_packets(blob: bytes | memoryview, packets: list[bytes]) -> None:
    # rage opus streams split into 2048 byte blocks of u16 prefixed frames;
    # a 320-byte frame leaves room for ~6 more in the same block, so this
    # has to walk every packet in the block, not just the first
    mv = memoryview(blob)
    n = len(mv)
    for bpos in range(0, n, 2048):
        bend = min(bpos + 2048, n)
        p = bpos
        while p + 2 <= bend:
            plen = struct.unpack_from("<H", mv, p)[0]
            if plen == 0 or p + 2 + plen > bend:
                break
            packets.append(bytes(mv[p + 2 : p + 2 + plen]))
            p += 2 + plen


def packets_for_superblock_substream(blob: bytes, n_sub: int, sub_idx: int) -> list[bytes]:
    """walks each 0x80000 (512kb) superblock, skips its directory, and collects
    only the real packets belonging to one substream -- ported from awc.rs's
    decode_opus_chunk_into so the python tool quits reading directory bytes
    (0xFFFFFFFF markers + per-block counts) as if they were opus packets."""
    packets: list[bytes] = []
    for sb_start in range(0, len(blob), 0x80000):
        sb = blob[sb_start : sb_start + 0x80000]
        if len(sb) < n_sub * 24:
            continue
        counts = []
        valid = True
        total_blks = 0
        for s in range(n_sub):
            if ru32(sb, s * 24) != 0xFFFFFFFF:
                valid = False
                break
            c = ru32(sb, s * 24 + 4)
            counts.append(c)
            total_blks += c
        if not valid or total_blks == 0 or total_blks > 256:
            continue
        dir_bytes = n_sub * 24 + total_blks * 4
        dir_blks = max(1, (dir_bytes + 2047) // 2048)
        start_blk = dir_blks + sum(counts[:sub_idx])
        end_blk = start_blk + counts[sub_idx]
        byte_start, byte_end = start_blk * 2048, min(end_blk * 2048, len(sb))
        if byte_start < byte_end:
            collect_2kb_packets(sb[byte_start:byte_end], packets)
    return packets


def channels_from_packets(packets: list[bytes]) -> int:
    # opus toc byte bit 0x04 (the 's' bit, RFC 6716 3.1) signals stereo vs mono;
    # the OpusHead channel-count field (RFC 7845 5.1) must match or ffmpeg
    # mis-decodes the stream instead of producing a correct stereo wav.
    # go by the LARGEST packet, not packets[0]: real streams often open on a
    # tiny 2-3 byte DTX/silence frame, and any leftover directory-table byte
    # that slips through would also land at index 0 -- a stray single-byte
    # misread there flips the whole track to mono even when every real frame
    # (and the DTX frame itself, going by its own TOC byte) says stereo.
    if not packets:
        return 1
    biggest = max(packets, key=len)
    return 2 if (biggest[0] & 0x04) else 1


def extract_awc(awc_path: str, out_dir: str) -> list[str]:
    with open(awc_path, "rb") as f:
        data = f.read()
    if len(data) < 16 or data[:4] != b"ADAT":
        return []
    ver, flags, sc, hdr_sz = struct.unpack_from("<HHII", data, 4)
    if sc == 0 or sc > 4096:
        return []

    chunk_table_start = 16 + sc * 4
    off = chunk_table_start
    data_chunks = []
    # unpack 64bit tag size offset entries lookign for data chunks 0x55
    while off + 8 <= len(data) and off < hdr_sz:
        val = struct.unpack_from("<Q", data, off)[0]
        tag = val >> 56
        sz = (val >> 28) & 0x0FFFFFFF
        coff = val & 0x0FFFFFFF
        if tag == 0x55 and 0 < sz <= len(data) - coff and coff >= 0x20:
            data_chunks.append((coff, sz))
        if tag == 0 and coff == 0:
            break
        off += 8

    base = os.path.splitext(os.path.basename(awc_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_wavs = []
    for idx, (coff, csz) in enumerate(data_chunks):
        blob = data[coff : coff + csz]

        # 512kb superblock-packed chunks (multi-channel/multi-stream banks) put
        # a block-count directory (0xFFFFFFFF marker + counts) at the start of
        # each superblock; reading it flat as [u16 len][payload] packets (like
        # the old code here did) feeds directory bytes into the opus stream as
        # bogus packets, and since channel detection only ever looked at the
        # FIRST packet, a single misread byte there was enough to silently
        # flip an actual stereo track to mono. awc.rs already does this split
        # correctly (probe_superblock_streams / decode_opus_chunk_into) -- this
        # mirrors it so the two no longer disagree on the same file.
        n_sub = probe_superblock_streams(blob)
        substreams: list[list[bytes]]
        if n_sub is None:
            packets: list[bytes] = []
            collect_2kb_packets(blob, packets)
            substreams = [packets]
        else:
            substreams = [packets_for_superblock_substream(blob, n_sub, s) for s in range(n_sub)]

        for sub_idx, packets in enumerate(substreams):
            if not packets:
                continue

            channels = channels_from_packets(packets)

            suffix = f"s{idx:02d}" if len(substreams) == 1 else f"s{idx:02d}_{sub_idx}"
            ogg_p = os.path.join(out_dir, f"{base}_{suffix}.opus")
            wav_p = os.path.join(out_dir, f"{base}_{suffix}.wav")
            serial = 0x50530001 + idx * 8 + sub_idx
            opus_head = struct.pack("<8sBBHIhB", b"OpusHead", 1, channels, 0, 48000, 0, 0)
            vendor = b"PercyStar RPF9 AWC"
            opus_tags = b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)
            ogg = bytearray()
            ogg.extend(ogg_page(serial, 0, 2, 0, [opus_head]))
            ogg.extend(ogg_page(serial, 1, 0, 0, [opus_tags]))
            granule = 0
            seq = 2
            for i in range(0, len(packets), 50):
                batch = packets[i : i + 50]
                granule += len(batch) * 960
                ht = 4 if (i + 50 >= len(packets)) else 0
                ogg.extend(ogg_page(serial, seq, ht, granule, batch))
                seq += 1
            with open(ogg_p, "wb") as wf:
                wf.write(ogg)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", ogg_p, wav_p], check=False)
            try:
                os.remove(ogg_p)
            except OSError:
                pass
            if os.path.exists(wav_p) and os.path.getsize(wav_p) > 44:
                out_wavs.append(wav_p)
    return out_wavs


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python awc_extract.py <input_awc_dir> <output_wav_dir>")
        sys.exit(1)
    in_dir, out_dir = sys.argv[1], sys.argv[2]
    total = 0
    for p in sorted(glob.glob(os.path.join(in_dir, "*.awc"))):
        total += len(extract_awc(p, out_dir))
    print(f"Done: {total} playable .wav files extracted to {out_dir}")
