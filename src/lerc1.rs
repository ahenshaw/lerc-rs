use crate::{
    bitmask::BitMask,
    bitstuffer,
    error::LercError,
    rle::rle_decompress,
    types::{DataType, DecodedData, LercData, LercInfo},
};

pub(crate) const MAGIC: &[u8] = b"CntZImage ";
const VERSION: i32 = 11;
const TYPE_CNT_Z: i32 = 8;

pub(crate) fn is_lerc1(src: &[u8]) -> bool {
    src.starts_with(MAGIC)
}

// ── Read helpers ──────────────────────────────────────────────────────────────

fn read_i32(src: &[u8], pos: &mut usize) -> Result<i32, LercError> {
    if *pos + 4 > src.len() {
        return Err(LercError::TruncatedBlob);
    }
    let v = i32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn read_f32(src: &[u8], pos: &mut usize) -> Result<f32, LercError> {
    if *pos + 4 > src.len() {
        return Err(LercError::TruncatedBlob);
    }
    let v = f32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn read_f64(src: &[u8], pos: &mut usize) -> Result<f64, LercError> {
    if *pos + 8 > src.len() {
        return Err(LercError::TruncatedBlob);
    }
    let v = f64::from_le_bytes(src[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

/// Read a float stored in n bytes: 1 = i8, 2 = i16 LE, 4 = f32 LE.
fn read_flt(src: &[u8], pos: &mut usize, n: usize) -> Result<f32, LercError> {
    match n {
        1 => {
            let c = *src.get(*pos).ok_or(LercError::TruncatedBlob)? as i8;
            *pos += 1;
            Ok(c as f32)
        }
        2 => {
            if *pos + 2 > src.len() {
                return Err(LercError::TruncatedBlob);
            }
            let s = i16::from_le_bytes(src[*pos..*pos + 2].try_into().unwrap());
            *pos += 2;
            Ok(s as f32)
        }
        4 => read_f32(src, pos),
        _ => Err(LercError::InvalidBlob),
    }
}

fn read_uint_n(src: &[u8], pos: &mut usize, n: usize) -> Result<u32, LercError> {
    match n {
        1 => {
            let v = *src.get(*pos).ok_or(LercError::TruncatedBlob)?;
            *pos += 1;
            Ok(v as u32)
        }
        2 => {
            if *pos + 2 > src.len() {
                return Err(LercError::TruncatedBlob);
            }
            let v = u16::from_le_bytes(src[*pos..*pos + 2].try_into().unwrap());
            *pos += 2;
            Ok(v as u32)
        }
        4 => {
            if *pos + 4 > src.len() {
                return Err(LercError::TruncatedBlob);
            }
            let v = u32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Ok(v)
        }
        _ => Err(LercError::InvalidBlob),
    }
}

// ── Lerc1 BitStuffer ──────────────────────────────────────────────────────────

/// Parse the Lerc1 BitStuffer2 header and decode the packed integers.
///
/// Header byte: bits[7:6] = size of numElements field, bits[5:0] = numBits.
/// Uses the same MSB-first unpacking as the Lerc2 pre-v3 bit stuffer.
fn read_bitstuffer(src: &[u8], pos: &mut usize) -> Result<Vec<u32>, LercError> {
    let hdr = *src.get(*pos).ok_or(LercError::TruncatedBlob)?;
    *pos += 1;

    let bits67 = (hdr >> 6) as usize;
    let n = if bits67 == 0 { 4 } else { 3 - bits67 };
    let num_bits = (hdr & 63) as usize;

    if num_bits >= 32 {
        return Err(LercError::InvalidBlob);
    }

    let num_elements = read_uint_n(src, pos, n)? as usize;
    let mut out = vec![0u32; num_elements];

    if num_elements > 0 && num_bits > 0 {
        if !bitstuffer::bit_unstuff_pre_v3(src, pos, num_elements, num_bits, &mut out) {
            return Err(LercError::TruncatedBlob);
        }
    }

    Ok(out)
}

// ── Header ────────────────────────────────────────────────────────────────────

struct Header {
    width: usize,
    height: usize,
    max_z_error: f64,
}

fn parse_header(src: &[u8], pos: &mut usize) -> Result<Header, LercError> {
    if !src.starts_with(MAGIC) {
        return Err(LercError::InvalidBlob);
    }
    *pos = MAGIC.len();

    let version = read_i32(src, pos)?;
    let type_ = read_i32(src, pos)?;
    let height = read_i32(src, pos)?;
    let width = read_i32(src, pos)?;
    let max_z_error = read_f64(src, pos)?;

    if version != VERSION || type_ != TYPE_CNT_Z {
        return Err(LercError::InvalidBlob);
    }
    if width <= 0 || height <= 0 || width > 20000 || height > 20000 {
        return Err(LercError::InvalidBlob);
    }

    Ok(Header { width: width as usize, height: height as usize, max_z_error })
}

struct PartHeader {
    num_tiles_vert: usize,
    num_tiles_hori: usize,
    num_bytes: usize,
    max_val: f32,
}

fn read_part_header(src: &[u8], pos: &mut usize) -> Result<PartHeader, LercError> {
    let ntv = read_i32(src, pos)?;
    let nth = read_i32(src, pos)?;
    let nb = read_i32(src, pos)?;
    let mv = read_f32(src, pos)?;

    if ntv < 0 || nth < 0 || nb < 0 {
        return Err(LercError::InvalidBlob);
    }

    Ok(PartHeader {
        num_tiles_vert: ntv as usize,
        num_tiles_hori: nth as usize,
        num_bytes: nb as usize,
        max_val: mv,
    })
}

// ── Cnt tile ──────────────────────────────────────────────────────────────────

fn read_cnt_tile(
    src: &[u8],
    pos: &mut usize,
    width: usize,
    mask: &mut [u8],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) -> Result<(), LercError> {
    let compr_flag = *src.get(*pos).ok_or(LercError::TruncatedBlob)?;
    *pos += 1;

    // Constant-tile fast paths.
    if compr_flag == 2 {
        // All pixels invalid (cnt = 0); mask is already 0 from initialization.
        return Ok(());
    }
    if compr_flag == 3 {
        // cnt = -1 → invalid.
        for i in i0..i1 {
            mask[i * width + j0..i * width + j1].fill(0);
        }
        return Ok(());
    }
    if compr_flag == 4 {
        // cnt = 1 → valid.
        for i in i0..i1 {
            mask[i * width + j0..i * width + j1].fill(1);
        }
        return Ok(());
    }

    if (compr_flag & 63) > 4 {
        return Err(LercError::InvalidBlob);
    }

    if compr_flag == 0 {
        // Raw f32 per pixel.
        for i in i0..i1 {
            for j in j0..j1 {
                let cnt = read_f32(src, pos)?;
                mask[i * width + j] = u8::from(cnt > 0.0);
            }
        }
    } else {
        // Bit-stuffed with float offset. bits[7:6] of compr_flag encode offset width.
        let bits67 = (compr_flag >> 6) as usize;
        let n = if bits67 == 0 { 4 } else { 3 - bits67 };
        let offset = read_flt(src, pos, n)?;
        let data = read_bitstuffer(src, pos)?;

        let tile_size = (i1 - i0) * (j1 - j0);
        if data.len() < tile_size {
            return Err(LercError::DecodeFailed);
        }

        let mut idx = 0;
        for i in i0..i1 {
            for j in j0..j1 {
                let cnt = offset + data[idx] as f32;
                mask[i * width + j] = u8::from(cnt > 0.0);
                idx += 1;
            }
        }
    }

    Ok(())
}

// ── Z tile ────────────────────────────────────────────────────────────────────

fn read_z_tile(
    src: &[u8],
    pos: &mut usize,
    width: usize,
    z: &mut [f32],
    mask: &[u8],
    can_ignore_mask: bool,
    max_z_error: f64,
    max_z_val: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) -> Result<(), LercError> {
    let raw_flag = *src.get(*pos).ok_or(LercError::TruncatedBlob)?;
    *pos += 1;
    let bits67 = (raw_flag >> 6) as usize;
    let compr_flag = raw_flag & 63;

    if compr_flag == 2 {
        // All valid pixels get z = 0.
        for i in i0..i1 {
            for j in j0..j1 {
                if mask[i * width + j] > 0 {
                    z[i * width + j] = 0.0;
                }
            }
        }
        return Ok(());
    }

    if compr_flag > 3 {
        return Err(LercError::InvalidBlob);
    }

    if compr_flag == 0 {
        // Raw f32 for valid pixels only.
        for i in i0..i1 {
            for j in j0..j1 {
                if mask[i * width + j] > 0 {
                    z[i * width + j] = read_f32(src, pos)?;
                }
            }
        }
        return Ok(());
    }

    // compr_flag == 1 (bit-stuffed) or 3 (constant).
    let n = if bits67 == 0 { 4 } else { 3 - bits67 };
    let offset = read_flt(src, pos, n)?;

    if compr_flag == 3 {
        // Constant: all valid pixels get z = offset.
        for i in i0..i1 {
            for j in j0..j1 {
                if mask[i * width + j] > 0 {
                    z[i * width + j] = offset;
                }
            }
        }
        return Ok(());
    }

    // Bit-stuffed (compr_flag == 1).
    let data = read_bitstuffer(src, pos)?;
    let inv_scale = 2.0 * max_z_error;
    let mut src_idx = 0usize;

    if can_ignore_mask {
        for i in i0..i1 {
            for j in j0..j1 {
                let val = *data.get(src_idx).ok_or(LercError::DecodeFailed)?;
                src_idx += 1;
                let zv = (offset as f64 + val as f64 * inv_scale) as f32;
                z[i * width + j] = zv.min(max_z_val);
            }
        }
    } else {
        for i in i0..i1 {
            for j in j0..j1 {
                if mask[i * width + j] > 0 {
                    let val = *data.get(src_idx).ok_or(LercError::DecodeFailed)?;
                    src_idx += 1;
                    let zv = (offset as f64 + val as f64 * inv_scale) as f32;
                    z[i * width + j] = zv.min(max_z_val);
                }
            }
        }
    }

    Ok(())
}

// ── Tile loops ────────────────────────────────────────────────────────────────

fn tile_bounds(
    n: usize,
    num_tiles: usize,
    i_tile: usize,
) -> (usize, usize) {
    let base = n / num_tiles;
    let i0 = i_tile * base;
    let size = if i_tile == num_tiles { n % num_tiles } else { base };
    (i0, i0 + size)
}

fn decode_cnt_tiles(
    src: &[u8],
    pos: &mut usize,
    width: usize,
    height: usize,
    num_tiles_vert: usize,
    num_tiles_hori: usize,
    mask: &mut Vec<u8>,
) -> Result<(), LercError> {
    if num_tiles_vert == 0 || num_tiles_hori == 0 {
        return Err(LercError::InvalidBlob);
    }

    for i_tile in 0..=num_tiles_vert {
        let (i0, i1) = tile_bounds(height, num_tiles_vert, i_tile);
        if i0 == i1 {
            continue;
        }
        for j_tile in 0..=num_tiles_hori {
            let (j0, j1) = tile_bounds(width, num_tiles_hori, j_tile);
            if j0 == j1 {
                continue;
            }
            read_cnt_tile(src, pos, width, mask, i0, i1, j0, j1)?;
        }
    }

    Ok(())
}

fn decode_z_tiles(
    src: &[u8],
    pos: &mut usize,
    width: usize,
    height: usize,
    num_tiles_vert: usize,
    num_tiles_hori: usize,
    max_z_error: f64,
    max_z_val: f32,
    mask: &[u8],
    can_ignore_mask: bool,
    z: &mut Vec<f32>,
) -> Result<(), LercError> {
    if num_tiles_vert == 0 || num_tiles_hori == 0 {
        return Err(LercError::InvalidBlob);
    }

    for i_tile in 0..=num_tiles_vert {
        let (i0, i1) = tile_bounds(height, num_tiles_vert, i_tile);
        if i0 == i1 {
            continue;
        }
        for j_tile in 0..=num_tiles_hori {
            let (j0, j1) = tile_bounds(width, num_tiles_hori, j_tile);
            if j0 == j1 {
                continue;
            }
            read_z_tile(
                src, pos, width, z, mask, can_ignore_mask,
                max_z_error, max_z_val, i0, i1, j0, j1,
            )?;
        }
    }

    Ok(())
}

// ── Cnt part ──────────────────────────────────────────────────────────────────

/// Decode the validity mask (cnt part).
/// Returns (mask, can_ignore_mask) where mask[i*width+j] = 1 if valid.
fn decode_cnt_part(
    src: &[u8],
    pos: &mut usize,
    width: usize,
    height: usize,
) -> Result<(Vec<u8>, bool), LercError> {
    let hdr = read_part_header(src, pos)?;
    let data_start = *pos;

    let mut mask = vec![0u8; width * height];
    let mut can_ignore_mask = false;

    if hdr.num_tiles_vert == 0 && hdr.num_tiles_hori == 0 {
        if hdr.num_bytes == 0 {
            // Constant cnt.
            let val = u8::from(hdr.max_val > 0.0);
            mask.fill(val);
            can_ignore_mask = hdr.max_val > 0.0;
        } else {
            // RLE-compressed BitMask.
            let compressed = src
                .get(data_start..data_start + hdr.num_bytes)
                .ok_or(LercError::TruncatedBlob)?;
            let mut bitmask =
                BitMask::new(width as i32, height as i32).ok_or(LercError::InvalidBlob)?;
            rle_decompress(compressed, bitmask.bits_mut())?;
            for k in 0..width * height {
                mask[k] = u8::from(bitmask.is_valid(k as i32));
            }
        }
    } else {
        decode_cnt_tiles(src, pos, width, height, hdr.num_tiles_vert, hdr.num_tiles_hori, &mut mask)?;
    }

    *pos = data_start + hdr.num_bytes;
    if *pos > src.len() {
        return Err(LercError::TruncatedBlob);
    }

    Ok((mask, can_ignore_mask))
}

// ── Z part ────────────────────────────────────────────────────────────────────

fn decode_z_part(
    src: &[u8],
    pos: &mut usize,
    width: usize,
    height: usize,
    max_z_error: f64,
    mask: &[u8],
    can_ignore_mask: bool,
) -> Result<Vec<f32>, LercError> {
    let hdr = read_part_header(src, pos)?;
    let data_start = *pos;

    let mut z = vec![0.0f32; width * height];

    if hdr.num_tiles_vert == 0 && hdr.num_tiles_hori == 0 {
        // No-tile z part: if numBytes == 0, all z values are 0 (already initialized).
        // Any other numBytes value with no-tile z is malformed.
        if hdr.num_bytes != 0 {
            return Err(LercError::InvalidBlob);
        }
    } else {
        decode_z_tiles(
            src, pos, width, height,
            hdr.num_tiles_vert, hdr.num_tiles_hori,
            max_z_error, hdr.max_val,
            mask, can_ignore_mask,
            &mut z,
        )?;
    }

    *pos = data_start + hdr.num_bytes;
    if *pos > src.len() {
        return Err(LercError::TruncatedBlob);
    }

    Ok(z)
}

// ── Public API ────────────────────────────────────────────────────────────────

pub(crate) fn decode(src: &[u8]) -> Result<DecodedData, LercError> {
    let mut pos = 0;
    let hdr = parse_header(src, &mut pos)?;

    let (mask, can_ignore_mask) = decode_cnt_part(src, &mut pos, hdr.width, hdr.height)?;
    let z = decode_z_part(src, &mut pos, hdr.width, hdr.height, hdr.max_z_error, &mask, can_ignore_mask)?;

    let n_pixels = hdr.width * hdr.height;
    let num_valid = mask.iter().filter(|&&b| b > 0).count() as i32;
    let all_valid = num_valid == n_pixels as i32;

    let (z_min, z_max) = if num_valid == 0 {
        (0.0f64, 0.0f64)
    } else if all_valid {
        let mn = z.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let mx = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        (mn, mx)
    } else {
        let mn = z.iter().zip(mask.iter()).filter(|&(_, &m)| m > 0).map(|(&v, _)| v).fold(f32::INFINITY, f32::min) as f64;
        let mx = z.iter().zip(mask.iter()).filter(|&(_, &m)| m > 0).map(|(&v, _)| v).fold(f32::NEG_INFINITY, f32::max) as f64;
        (mn, mx)
    };

    let info = LercInfo {
        version: 0,
        n_depth: 1,
        n_cols: hdr.width as i32,
        n_rows: hdr.height as i32,
        num_valid_pixel: num_valid,
        n_bands: 1,
        blob_size: src.len() as i32,
        n_masks: i32::from(!all_valid),
        n_uses_no_data_value: 0,
        data_type: DataType::F32,
        z_min,
        z_max,
        max_z_error: hdr.max_z_error,
    };

    Ok(DecodedData {
        data: LercData::F32(z),
        valid_pixels: if all_valid { None } else { Some(mask) },
        no_data_values: None,
        info,
    })
}

pub(crate) fn get_lerc_info(src: &[u8]) -> Result<LercInfo, LercError> {
    let mut pos = 0;
    let hdr = parse_header(src, &mut pos)?;

    let (mask, _) = decode_cnt_part(src, &mut pos, hdr.width, hdr.height)?;
    // Read z part header only to get z_max; skip tile data.
    let z_hdr = read_part_header(src, &mut pos)?;

    let num_valid = mask.iter().filter(|&&b| b > 0).count() as i32;
    let n_pixels = (hdr.width * hdr.height) as i32;

    Ok(LercInfo {
        version: 0,
        n_depth: 1,
        n_cols: hdr.width as i32,
        n_rows: hdr.height as i32,
        num_valid_pixel: num_valid,
        n_bands: 1,
        blob_size: src.len() as i32,
        n_masks: i32::from(num_valid < n_pixels),
        n_uses_no_data_value: 0,
        data_type: DataType::F32,
        z_min: 0.0,
        z_max: z_hdr.max_val as f64,
        max_z_error: hdr.max_z_error,
    })
}
