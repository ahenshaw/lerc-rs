/// LERC2 single-band decoder.
///
/// Implements get_lerc_info and decode for Lerc2 v1–v6 blobs (multi-band via
/// concatenated single-band blobs).  Lerc1 (CntZImage) is not supported.
use crate::{
    bitmask::BitMask,
    bitstuffer,
    error::LercError,
    huffman::HuffmanDecoder,
    lossless_float,
    rle,
    types::{DataType, DecodedData, LercData, LercInfo},
};

// ── Constants ────────────────────────────────────────────────────────────────

const MAGIC: &[u8] = b"Lerc2 ";
const KEY_LEN: usize = 6;

// ── Internal header struct ───────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
struct HeaderInfo {
    version: i32,
    checksum: u32,
    n_rows: i32,
    n_cols: i32,
    n_depth: i32,
    num_valid_pixel: i32,
    micro_block_size: i32,
    blob_size: i32,
    dt: DataType,
    n_blobs_more: i32,
    b_pass_no_data_values: bool,
    max_z_error: f64,
    z_min: f64,
    z_max: f64,
    no_data_val: f64,
    no_data_val_orig: f64,
}

impl HeaderInfo {
    fn try_huffman_int(&self) -> bool {
        self.version >= 2
            && matches!(self.dt, DataType::U8 | DataType::I8)
            && (self.max_z_error - 0.5).abs() < 1e-10
    }
    fn try_huffman_flt(&self) -> bool {
        self.version >= 6
            && matches!(self.dt, DataType::F32 | DataType::F64)
            && self.max_z_error == 0.0
    }
}

impl Default for DataType {
    fn default() -> Self {
        DataType::U8
    }
}

// ── ImageEncodeMode ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum ImageEncodeMode {
    Tiling = 0,
    DeltaHuffman = 1,
    Huffman = 2,
    DeltaDeltaHuffman = 3,
}

// ── Byte-level read helpers ──────────────────────────────────────────────────

#[inline]
fn read_i32(src: &[u8], pos: &mut usize) -> Result<i32, LercError> {
    if *pos + 4 > src.len() {
        return Err(LercError::TruncatedBlob);
    }
    let v = i32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

#[inline]
fn read_u32(src: &[u8], pos: &mut usize) -> Result<u32, LercError> {
    if *pos + 4 > src.len() {
        return Err(LercError::TruncatedBlob);
    }
    let v = u32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

#[inline]
fn read_f64(src: &[u8], pos: &mut usize) -> Result<f64, LercError> {
    if *pos + 8 > src.len() {
        return Err(LercError::TruncatedBlob);
    }
    let v = f64::from_le_bytes(src[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

#[inline]
fn read_u8(src: &[u8], pos: &mut usize) -> Result<u8, LercError> {
    if *pos >= src.len() {
        return Err(LercError::TruncatedBlob);
    }
    let v = src[*pos];
    *pos += 1;
    Ok(v)
}

// ── Header parsing ───────────────────────────────────────────────────────────

fn read_header(src: &[u8], pos: &mut usize) -> Result<HeaderInfo, LercError> {
    if src.len() < KEY_LEN + 4 {
        return Err(LercError::TruncatedBlob);
    }
    if &src[..KEY_LEN] != MAGIC {
        return Err(LercError::InvalidBlob);
    }
    *pos = KEY_LEN;

    let version = read_i32(src, pos)?;
    if version < 1 || version > 6 {
        return Err(LercError::UnsupportedVersion(version));
    }

    let checksum = if version >= 3 { read_u32(src, pos)? } else { 0 };

    let n_rows = read_i32(src, pos)?;
    let n_cols = read_i32(src, pos)?;
    let n_depth = if version >= 4 { read_i32(src, pos)? } else { 1 };
    let num_valid_pixel = read_i32(src, pos)?;
    let micro_block_size = read_i32(src, pos)?;
    let blob_size = read_i32(src, pos)?;
    let dt_i = read_i32(src, pos)?;
    let dt = DataType::from_i32(dt_i).ok_or(LercError::InvalidBlob)?;

    let n_blobs_more = if version >= 6 { read_i32(src, pos)? } else { 0 };

    let b_pass_no_data_values = if version >= 6 {
        let b1 = read_u8(src, pos)? != 0;
        let _b2 = read_u8(src, pos)?;
        let _b3 = read_u8(src, pos)?;
        let _b4 = read_u8(src, pos)?;
        b1
    } else {
        false
    };

    let max_z_error = read_f64(src, pos)?;
    let z_min = read_f64(src, pos)?;
    let z_max = read_f64(src, pos)?;

    // v6+ always stores noDataVal and noDataValOrig in the header (16 bytes),
    // regardless of bPassNoDataValues.
    let (no_data_val, no_data_val_orig) = if version >= 6 {
        (read_f64(src, pos)?, read_f64(src, pos)?)
    } else {
        (0.0, 0.0)
    };

    if n_rows <= 0
        || n_cols <= 0
        || n_depth <= 0
        || num_valid_pixel < 0
        || micro_block_size <= 0
        || blob_size <= 0
        || num_valid_pixel > n_rows * n_cols
    {
        return Err(LercError::InvalidBlob);
    }

    Ok(HeaderInfo {
        version,
        checksum,
        n_rows,
        n_cols,
        n_depth,
        num_valid_pixel,
        micro_block_size,
        blob_size,
        dt,
        n_blobs_more,
        b_pass_no_data_values,
        max_z_error,
        z_min,
        z_max,
        no_data_val,
        no_data_val_orig,
    })
}

/// Try to read the Lerc2 header from `src[base..]` without updating a position
/// cursor.  Returns (HeaderInfo, has_mask) on success.
fn get_header_info(src: &[u8], base: usize) -> Option<(HeaderInfo, bool)> {
    if base >= src.len() {
        return None;
    }
    let slice = &src[base..];
    let mut pos = 0;
    let hd = read_header(slice, &mut pos).ok()?;
    // Peek at the mask byte count to determine bHasMask.
    if pos + 4 > slice.len() {
        return None;
    }
    let num_bytes_mask = i32::from_le_bytes(slice[pos..pos + 4].try_into().unwrap());
    let has_mask = num_bytes_mask > 0;
    Some((hd, has_mask))
}

// ── Fletcher-32 checksum ─────────────────────────────────────────────────────

fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0xffff;
    let mut sum2: u32 = 0xffff;
    let mut words = data.len() / 2;
    let mut ptr = 0usize;

    while words > 0 {
        let tlen = words.min(359);
        words -= tlen;
        for _ in 0..tlen {
            sum1 = sum1.wrapping_add((data[ptr] as u32) << 8);
            sum1 = sum1.wrapping_add(data[ptr + 1] as u32);
            sum2 = sum2.wrapping_add(sum1);
            ptr += 2;
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }
    // Straggler byte: mirrors C++ `sum2 += sum1 += *pByte << 8`.
    if data.len() & 1 != 0 {
        sum1 = sum1.wrapping_add((data[ptr] as u32) << 8);
        sum2 = sum2.wrapping_add(sum1);
    }
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    sum2 << 16 | sum1
}

// ── GetDataTypeUsed ──────────────────────────────────────────────────────────

/// Map (source DataType, reduction code) → type used to store the tile offset.
///
/// Mirrors C++ `Lerc2::GetDataTypeUsed`.  Returns None for out-of-range dt.
fn get_data_type_used(dt: DataType, tc: i32) -> Option<DataType> {
    match dt {
        // Signed integers: subtract tc from enum index.
        DataType::I16 | DataType::I32 => DataType::from_i32(dt as i32 - tc),
        // Unsigned integers: subtract 2*tc from enum index.
        DataType::U16 | DataType::U32 => DataType::from_i32(dt as i32 - 2 * tc),
        // Float32 hard-coded reductions.
        DataType::F32 => match tc {
            0 => Some(DataType::F32),
            1 => Some(DataType::I16),
            _ => Some(DataType::U8),
        },
        // Float64: dt - 2*tc + 1.
        DataType::F64 => match tc {
            0 => Some(DataType::F64),
            _ => DataType::from_i32(dt as i32 - 2 * tc + 1),
        },
        // I8, U8: no reduction.
        _ => Some(dt),
    }
}

// ── ReadVariableDataType ─────────────────────────────────────────────────────

fn read_variable_data_type(
    src: &[u8],
    pos: &mut usize,
    dt: DataType,
) -> Result<f64, LercError> {
    Ok(match dt {
        DataType::I8 => {
            let v = src.get(*pos).copied().ok_or(LercError::TruncatedBlob)? as i8;
            *pos += 1;
            v as f64
        }
        DataType::U8 => {
            let v = src.get(*pos).copied().ok_or(LercError::TruncatedBlob)?;
            *pos += 1;
            v as f64
        }
        DataType::I16 => {
            if *pos + 2 > src.len() {
                return Err(LercError::TruncatedBlob);
            }
            let v = i16::from_le_bytes(src[*pos..*pos + 2].try_into().unwrap());
            *pos += 2;
            v as f64
        }
        DataType::U16 => {
            if *pos + 2 > src.len() {
                return Err(LercError::TruncatedBlob);
            }
            let v = u16::from_le_bytes(src[*pos..*pos + 2].try_into().unwrap());
            *pos += 2;
            v as f64
        }
        DataType::I32 => {
            let v = read_i32(src, pos)?;
            v as f64
        }
        DataType::U32 => {
            let v = read_u32(src, pos)?;
            v as f64
        }
        DataType::F32 => {
            if *pos + 4 > src.len() {
                return Err(LercError::TruncatedBlob);
            }
            let v = f32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            v as f64
        }
        DataType::F64 => read_f64(src, pos)?,
    })
}

// ── LercScalar trait ─────────────────────────────────────────────────────────

pub(crate) trait LercScalar: Copy + Default + PartialEq + 'static {
    /// C-style cast from f64 (truncation for integers, direct for floats).
    fn cast_f64(v: f64) -> Self;
    /// Lossless conversion to f64.
    fn to_f64(self) -> f64;
    /// Read `sizeof(Self)` bytes as little-endian from `src[*pos..]`.
    fn read_le(src: &[u8], pos: &mut usize) -> Result<Self, LercError>;
    /// Wrapping add of `self` and `other` (used in Huffman delta decode).
    fn wrapping_add(self, other: Self) -> Self;
    /// Wrapping cast from i32 (for Huffman: symbol - offset).
    fn from_i32_wrapping(v: i32) -> Self;

    /// Dequantize a contiguous slice of u32 codes into `out`:
    ///   `out[i] = (buf[i] as Self * inv_scale + offset).min(z_max)`
    ///
    /// The default is a scalar loop via f64.  Float types override this
    /// with SIMD-accelerated implementations.
    #[inline]
    fn dequantize_slice(buf: &[u32], out: &mut [Self], offset: f64, inv_scale: f64, z_max: f64) {
        for (q, o) in buf.iter().zip(out.iter_mut()) {
            *o = Self::cast_f64((offset + *q as f64 * inv_scale).min(z_max));
        }
    }

    /// Decode a DeltaDeltaHuffman (lossless float) encoded block.
    /// Only implemented for f32 and f64; all other types return UnsupportedFeature.
    fn decode_lossless_flt(
        _src: &[u8],
        _pos: &mut usize,
        _data: &mut [Self],
        _n_cols: usize,
        _n_rows: usize,
        _n_depth: usize,
    ) -> Result<(), LercError> {
        Err(LercError::UnsupportedFeature("lossless float (DeltaDeltaHuffman)"))
    }
}

macro_rules! impl_lerc_scalar_int {
    ($t:ty, $variant:ident) => {
        impl LercScalar for $t {
            #[inline] fn cast_f64(v: f64) -> Self { v as $t }
            #[inline] fn to_f64(self) -> f64 { self as f64 }
            fn read_le(src: &[u8], pos: &mut usize) -> Result<Self, LercError> {
                const SZ: usize = core::mem::size_of::<$t>();
                if *pos + SZ > src.len() { return Err(LercError::TruncatedBlob); }
                let arr: [u8; SZ] = src[*pos..*pos + SZ].try_into().unwrap();
                *pos += SZ;
                Ok(<$t>::from_le_bytes(arr))
            }
            #[inline] fn wrapping_add(self, other: Self) -> Self { self.wrapping_add(other) }
            #[inline] fn from_i32_wrapping(v: i32) -> Self { v as $t }
        }
    };
}

macro_rules! impl_lerc_scalar_flt {
    ($t:ty, $variant:ident, $dequant:expr, $lossless:expr) => {
        impl LercScalar for $t {
            #[inline] fn cast_f64(v: f64) -> Self { v as $t }
            #[inline] fn to_f64(self) -> f64 { self as f64 }
            fn read_le(src: &[u8], pos: &mut usize) -> Result<Self, LercError> {
                const SZ: usize = core::mem::size_of::<$t>();
                if *pos + SZ > src.len() { return Err(LercError::TruncatedBlob); }
                let arr: [u8; SZ] = src[*pos..*pos + SZ].try_into().unwrap();
                *pos += SZ;
                Ok(<$t>::from_le_bytes(arr))
            }
            #[inline] fn wrapping_add(self, other: Self) -> Self { self + other }
            #[inline] fn from_i32_wrapping(v: i32) -> Self { v as $t }
            #[inline]
            fn dequantize_slice(buf: &[u32], out: &mut [Self], offset: f64, inv_scale: f64, z_max: f64) {
                $dequant(buf, out, offset, inv_scale, z_max);
            }
            fn decode_lossless_flt(
                src: &[u8], pos: &mut usize, data: &mut [Self],
                n_cols: usize, n_rows: usize, n_depth: usize,
            ) -> Result<(), LercError> {
                $lossless(src, pos, data, n_cols, n_rows, n_depth)
            }
        }
    };
}

impl_lerc_scalar_int!(i8,  I8);
impl_lerc_scalar_int!(u8,  U8);
impl_lerc_scalar_int!(i16, I16);
impl_lerc_scalar_int!(u16, U16);
impl_lerc_scalar_int!(i32, I32);
impl_lerc_scalar_int!(u32, U32);
impl_lerc_scalar_flt!(f32, F32, crate::simd::dequantize_f32,
    lossless_float::decode_lossless_f32);
impl_lerc_scalar_flt!(f64, F64, crate::simd::dequantize_f64,
    lossless_float::decode_lossless_f64);

// ── ReadMask ─────────────────────────────────────────────────────────────────

fn read_mask(
    src: &[u8],
    pos: &mut usize,
    n_rows: i32,
    n_cols: i32,
    num_valid: i32,
) -> Result<BitMask, LercError> {
    if *pos + 4 > src.len() {
        return Err(LercError::TruncatedBlob);
    }
    let num_bytes_mask = i32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;

    let n_pixels = n_rows * n_cols;
    // Consistency: if all valid or all invalid, numBytesMask must be 0.
    if (num_valid == 0 || num_valid == n_pixels) && num_bytes_mask != 0 {
        return Err(LercError::InvalidBlob);
    }

    let mut mask = BitMask::new(n_cols, n_rows).ok_or(LercError::InvalidBlob)?;

    if num_valid == 0 {
        mask.set_all_invalid();
    } else if num_valid == n_pixels {
        mask.set_all_valid();
    } else if num_bytes_mask > 0 {
        let nb = num_bytes_mask as usize;
        if *pos + nb > src.len() {
            return Err(LercError::TruncatedBlob);
        }
        rle::rle_decompress(&src[*pos..*pos + nb], mask.bits_mut())?;
        *pos += nb;
    }
    // else: num_bytes_mask == 0 but not all-valid/invalid → retain previous
    // mask (not used in single-band decoding; the caller supplies fresh masks).

    Ok(mask)
}

// ── ReadMinMaxRanges ─────────────────────────────────────────────────────────

/// Read `n_depth` typed T values for mins, then `n_depth` for maxs.
fn read_min_max_ranges<T: LercScalar>(
    src: &[u8],
    pos: &mut usize,
    n_depth: usize,
) -> Result<(Vec<f64>, Vec<f64>), LercError> {
    let mut mins = Vec::with_capacity(n_depth);
    let mut maxs = Vec::with_capacity(n_depth);
    for _ in 0..n_depth {
        mins.push(T::read_le(src, pos)?.to_f64());
    }
    for _ in 0..n_depth {
        maxs.push(T::read_le(src, pos)?.to_f64());
    }
    Ok((mins, maxs))
}

// ── FillConstImage ───────────────────────────────────────────────────────────

/// Fill all valid pixels with z_min (n_depth==1) or per-depth values.
fn fill_const_image<T: LercScalar>(
    data: &mut [T],
    mask: &BitMask,
    n_rows: i32,
    n_cols: i32,
    n_depth: i32,
    z_min: f64,
    z_min_vec: &[f64], // empty → use scalar z_min for all depths
) {
    if n_depth == 1 {
        let val = T::cast_f64(z_min);
        for k in 0..(n_rows * n_cols) {
            if mask.is_valid(k) {
                data[k as usize] = val;
            }
        }
    } else {
        let buf: Vec<T> = if z_min_vec.len() == n_depth as usize {
            z_min_vec.iter().map(|&v| T::cast_f64(v)).collect()
        } else {
            vec![T::cast_f64(z_min); n_depth as usize]
        };
        for k in 0..(n_rows * n_cols) {
            if mask.is_valid(k) {
                let m = k as usize * n_depth as usize;
                data[m..m + n_depth as usize].copy_from_slice(&buf);
            }
        }
    }
}

// ── RemapNoData ──────────────────────────────────────────────────────────────

fn remap_no_data<T: LercScalar>(
    data: &mut [T],
    mask: &BitMask,
    n_rows: i32,
    n_cols: i32,
    n_depth: i32,
    no_data_old: T,
    no_data_new: T,
) {
    if no_data_old == no_data_new {
        return;
    }
    let nd = n_depth as usize;
    for k in 0..(n_rows * n_cols) as usize {
        if mask.is_valid(k as i32) {
            let m = k * nd;
            for d in 0..nd {
                if data[m + d] == no_data_old {
                    data[m + d] = no_data_new;
                }
            }
        }
    }
}

// ── ReadTile ─────────────────────────────────────────────────────────────────

fn read_tile<T: LercScalar>(
    src: &[u8],
    pos: &mut usize,
    data: &mut [T],
    i0: i32,
    i1: i32,
    j0: i32,
    j1: i32,
    i_depth: i32,
    hd: &HeaderInfo,
    mask: &BitMask,
    z_max_vec: &[f64], // per-depth zMax values (may be empty → use hd.z_max)
    buffer_vec: &mut Vec<u32>,
) -> Result<(), LercError> {
    if *pos >= src.len() {
        return Err(LercError::TruncatedBlob);
    }
    let compr_flag_raw = src[*pos];
    *pos += 1;

    let b_diff_enc = hd.version >= 5 && (compr_flag_raw & 4) != 0;
    let pattern: i32 = if hd.version >= 5 { 14 } else { 15 };

    // Pattern sanity check on the encoded column-block index.
    if ((compr_flag_raw as i32 >> 2) & pattern) != ((j0 >> 3) & pattern) {
        return Err(LercError::InvalidBlob);
    }
    if b_diff_enc && i_depth == 0 {
        return Err(LercError::InvalidBlob);
    }

    let bits67 = (compr_flag_raw >> 6) as i32;
    let compr_flag = compr_flag_raw & 3;

    let n_cols = hd.n_cols;
    let n_depth = hd.n_depth;

    match compr_flag {
        2 => {
            // Constant zero (or propagate from prev depth if diff enc).
            for i in i0..i1 {
                let k_base = i * n_cols + j0;
                let m_base = k_base * n_depth + i_depth;
                let mut k = k_base;
                let mut m = m_base;
                for _ in j0..j1 {
                    if mask.is_valid(k) {
                        data[m as usize] = if b_diff_enc {
                            data[(m - 1) as usize]
                        } else {
                            T::default()
                        };
                    }
                    k += 1;
                    m += n_depth;
                }
            }
        }

        0 => {
            // Raw binary (uncompressed); diff enc not allowed here.
            if b_diff_enc {
                return Err(LercError::InvalidBlob);
            }
            for i in i0..i1 {
                let k_base = i * n_cols + j0;
                let m_base = k_base * n_depth + i_depth;
                let mut k = k_base;
                let mut m = m_base;
                for _ in j0..j1 {
                    if mask.is_valid(k) {
                        let val = T::read_le(src, pos)?;
                        data[m as usize] = val;
                    }
                    k += 1;
                    m += n_depth;
                }
            }
        }

        1 | 3 => {
            // Bit-stuffed quantized values (1) or constant value (3).
            let eff_dt = if b_diff_enc && (hd.dt as i32) < (DataType::F32 as i32) {
                DataType::I32
            } else {
                hd.dt
            };
            let dt_used =
                get_data_type_used(eff_dt, bits67).ok_or(LercError::InvalidBlob)?;

            let offset = read_variable_data_type(src, pos, dt_used)?;

            let z_max = if hd.version >= 4
                && n_depth > 1
                && (i_depth as usize) < z_max_vec.len()
            {
                z_max_vec[i_depth as usize]
            } else {
                hd.z_max
            };

            if compr_flag == 3 {
                // Every valid pixel in the tile has the same value (= offset),
                // plus optional diff-from-prev-depth.
                for i in i0..i1 {
                    let k_base = i * n_cols + j0;
                    let m_base = k_base * n_depth + i_depth;
                    let mut k = k_base;
                    let mut m = m_base;
                    for _ in j0..j1 {
                        if mask.is_valid(k) {
                            data[m as usize] = if b_diff_enc {
                                let z = offset + data[(m - 1) as usize].to_f64();
                                T::cast_f64(z.min(z_max))
                            } else {
                                T::cast_f64(offset)
                            };
                        }
                        k += 1;
                        m += n_depth;
                    }
                }
            } else {
                // compr_flag == 1: decode bit-stuffed quantized values.
                let max_elem = ((i1 - i0) * (j1 - j0)) as usize;
                bitstuffer::decode(src, pos, max_elem, hd.version, buffer_vec)?;

                let inv_scale = 2.0 * hd.max_z_error;
                let all_valid = buffer_vec.len() == max_elem;
                let tile_w = (j1 - j0) as usize;

                if all_valid {
                    // Fast path: n_depth==1, no diff-encoding → row slices are
                    // contiguous in `data`, so we can dispatch to dequantize_slice
                    // which uses SIMD for float types.  chunks(tile_w) avoids the
                    // manual src_idx counter and its per-row bounds check.
                    if n_depth == 1 && !b_diff_enc {
                        for (chunk, i) in buffer_vec.chunks(tile_w).zip(i0..i1) {
                            let m = (i * n_cols + j0) as usize;
                            T::dequantize_slice(
                                chunk,
                                &mut data[m..m + tile_w],
                                offset,
                                inv_scale,
                                z_max,
                            );
                        }
                    } else {
                        // General all-valid path (n_depth > 1 or b_diff_enc).
                        let mut buf_iter = buffer_vec.iter();
                        for i in i0..i1 {
                            let k_base = i * n_cols + j0;
                            let m_base = k_base * n_depth + i_depth;
                            let mut m = m_base;
                            for _ in j0..j1 {
                                let &q = buf_iter.next().ok_or(LercError::DecodeFailed)?;
                                let z = offset + q as f64 * inv_scale;
                                data[m as usize] = if b_diff_enc {
                                    T::cast_f64((z + data[(m - 1) as usize].to_f64()).min(z_max))
                                } else {
                                    T::cast_f64(z.min(z_max))
                                };
                                m += n_depth;
                            }
                        }
                    }
                } else {
                    // Not all valid: buffer contains one value per valid pixel.
                    let mut buf_iter = buffer_vec.iter();
                    for i in i0..i1 {
                        let k_base = i * n_cols + j0;
                        let m_base = k_base * n_depth + i_depth;
                        let mut k = k_base;
                        let mut m = m_base;
                        for _ in j0..j1 {
                            if mask.is_valid(k) {
                                let &q = buf_iter.next().ok_or(LercError::DecodeFailed)?;
                                let z = offset + q as f64 * inv_scale;
                                data[m as usize] = if b_diff_enc {
                                    T::cast_f64((z + data[(m - 1) as usize].to_f64()).min(z_max))
                                } else {
                                    T::cast_f64(z.min(z_max))
                                };
                            }
                            k += 1;
                            m += n_depth;
                        }
                    }
                }
            }
        }

        _ => return Err(LercError::InvalidBlob),
    }

    Ok(())
}

// ── ReadTiles ─────────────────────────────────────────────────────────────────

fn read_tiles<T: LercScalar>(
    src: &[u8],
    pos: &mut usize,
    data: &mut [T],
    hd: &HeaderInfo,
    mask: &BitMask,
    z_max_vec: &[f64],
) -> Result<(), LercError> {
    let mb = hd.micro_block_size;
    if mb > 32 {
        return Err(LercError::InvalidBlob);
    }
    let n_tiles_vert = (hd.n_rows + mb - 1) / mb;
    let n_tiles_hori = (hd.n_cols + mb - 1) / mb;
    let mut buffer_vec: Vec<u32> = Vec::new();

    for i_tile in 0..n_tiles_vert {
        let i0 = i_tile * mb;
        let tile_h = if i_tile == n_tiles_vert - 1 { hd.n_rows - i0 } else { mb };

        for j_tile in 0..n_tiles_hori {
            let j0 = j_tile * mb;
            let tile_w =
                if j_tile == n_tiles_hori - 1 { hd.n_cols - j0 } else { mb };

            for i_depth in 0..hd.n_depth {
                read_tile(
                    src,
                    pos,
                    data,
                    i0,
                    i0 + tile_h,
                    j0,
                    j0 + tile_w,
                    i_depth,
                    hd,
                    mask,
                    z_max_vec,
                    &mut buffer_vec,
                )?;
            }
        }
    }
    Ok(())
}

// ── ReadDataOneSweep ──────────────────────────────────────────────────────────

fn read_data_one_sweep<T: LercScalar>(
    src: &[u8],
    pos: &mut usize,
    data: &mut [T],
    hd: &HeaderInfo,
    mask: &BitMask,
) -> Result<(), LercError> {
    let nd = hd.n_depth as usize;
    let n_valid = mask.count_valid_bits() as usize;
    let bytes_needed = n_valid * nd * core::mem::size_of::<T>();
    if *pos + bytes_needed > src.len() {
        return Err(LercError::TruncatedBlob);
    }
    for k in 0..(hd.n_rows * hd.n_cols) {
        if mask.is_valid(k) {
            let m = k as usize * nd;
            for d in 0..nd {
                data[m + d] = T::read_le(src, pos)?;
            }
        }
    }
    Ok(())
}

// ── DecodeHuffman ─────────────────────────────────────────────────────────────

fn decode_huffman<T: LercScalar>(
    src: &[u8],
    pos: &mut usize,
    data: &mut [T],
    hd: &HeaderInfo,
    mask: &BitMask,
    mode: ImageEncodeMode,
) -> Result<(), LercError> {
    let huff = HuffmanDecoder::from_blob(src, pos, hd.version)?;

    let offset: i32 = if hd.dt == DataType::I8 { 128 } else { 0 };
    let width = hd.n_cols;
    let height = hd.n_rows;
    let nd = hd.n_depth;
    let all_valid = hd.num_valid_pixel == width * height;

    let mut bit_pos: i32 = 0;

    match mode {
        ImageEncodeMode::DeltaHuffman => {
            for i_depth in 0..nd {
                let mut prev_val: T = T::default();
                for i in 0..height {
                    for j in 0..width {
                        let k = i * width + j;
                        let m = k * nd + i_depth;
                        if all_valid || mask.is_valid(k) {
                            let sym = if let Some(s) = huff.decode_one_fast(src, pos, &mut bit_pos) { s } else { huff.decode_one(src, pos, &mut bit_pos)? };
                            let delta = T::from_i32_wrapping(sym - offset);
                            // Mirror C++ logic exactly: if left neighbor exists and is
                            // valid, use prevVal; else if above neighbor exists and is
                            // valid, use that; else use prevVal.
                            let prev_nbr =
                                if j > 0 && (all_valid || mask.is_valid(k - 1)) {
                                    prev_val
                                } else if i > 0
                                    && (all_valid || mask.is_valid(k - width))
                                {
                                    data[(m - width * nd) as usize]
                                } else {
                                    prev_val
                                };
                            let v = delta.wrapping_add(prev_nbr);
                            data[m as usize] = v;
                            prev_val = v;
                        }
                    }
                }
            }
        }

        ImageEncodeMode::Huffman => {
            if all_valid && nd == 1 {
                // Dense sequential writes — iterate directly over output slice
                // to eliminate per-element index bounds checks.
                for out in data.iter_mut() {
                    let sym = if let Some(s) = huff.decode_one_fast(src, pos, &mut bit_pos) { s } else { huff.decode_one(src, pos, &mut bit_pos)? };
                    *out = T::from_i32_wrapping(sym - offset);
                }
            } else {
                for i in 0..height {
                    for j in 0..width {
                        let k = i * width + j;
                        let m0 = (k * nd) as usize;
                        if all_valid || mask.is_valid(k) {
                            for d in 0..nd as usize {
                                let sym = if let Some(s) = huff.decode_one_fast(src, pos, &mut bit_pos) { s } else { huff.decode_one(src, pos, &mut bit_pos)? };
                                data[m0 + d] = T::from_i32_wrapping(sym - offset);
                            }
                        }
                    }
                }
            }
        }

        _ => return Err(LercError::UnsupportedFeature("DeltaDeltaHuffman")),
    }

    // Advance pos past partial uint32 + 1 guard word.
    *pos += if bit_pos > 0 { 4 } else { 0 } + 4;
    Ok(())
}

// ── DecodeBandImpl ────────────────────────────────────────────────────────────

/// Decode one single-band Lerc2 blob into typed pixel data + bit-mask.
///
/// `src[blob_start..]` must begin with the magic bytes.
/// On success, `pos` is advanced to `blob_start + hd.blob_size`.
fn decode_band_impl<T: LercScalar>(
    src: &[u8],
    blob_start: usize,
) -> Result<(Vec<T>, BitMask, HeaderInfo), LercError> {
    let blob = &src[blob_start..];
    let mut pos = 0usize;

    let hd = read_header(blob, &mut pos)?;

    if blob_start + hd.blob_size as usize > src.len() {
        return Err(LercError::TruncatedBlob);
    }

    // Checksum verification (v3+).
    if hd.version >= 3 {
        // Covered region: from after the checksum field to blobSize.
        let skip = KEY_LEN + 4 + 4; // magic + version + checksum
        if (hd.blob_size as usize) < skip {
            return Err(LercError::InvalidBlob);
        }
        let csum = fletcher32(&blob[skip..hd.blob_size as usize]);
        if csum != hd.checksum {
            return Err(LercError::ChecksumMismatch);
        }
    }

    // Read bit-mask.
    let mask = read_mask(blob, &mut pos, hd.n_rows, hd.n_cols, hd.num_valid_pixel)?;

    // Allocate pixel buffer (zero-initialized).
    let n_total = hd.n_rows as usize * hd.n_cols as usize * hd.n_depth as usize;
    let mut data = vec![T::default(); n_total];

    if hd.num_valid_pixel == 0 {
        // All pixels invalid; buffer stays zero.
        return Ok((data, mask, hd));
    }

    // Constant image.
    if hd.z_min == hd.z_max {
        fill_const_image(&mut data, &mask, hd.n_rows, hd.n_cols, hd.n_depth, hd.z_min, &[]);
        return Ok((data, mask, hd));
    }

    // Per-depth min/max ranges (v4+): nDepth typed T values for mins, then nDepth for maxs.
    let (z_min_vec, z_max_vec) = if hd.version >= 4 {
        dispatch_read_min_max(blob, &mut pos, &hd)?
    } else {
        (Vec::new(), Vec::new())
    };

    // If all per-depth mins == maxs, fill constant.
    if !z_min_vec.is_empty()
        && z_min_vec.iter().zip(z_max_vec.iter()).all(|(a, b)| a == b)
    {
        fill_const_image(
            &mut data,
            &mask,
            hd.n_rows,
            hd.n_cols,
            hd.n_depth,
            hd.z_min,
            &z_min_vec,
        );
        return Ok((data, mask, hd));
    }

    // Read the "data one sweep" flag.
    let read_one_sweep = read_u8(blob, &mut pos)?;

    if read_one_sweep == 0 {
        // Check for Huffman encoding.
        if hd.try_huffman_int() || hd.try_huffman_flt() {
            let flag = read_u8(blob, &mut pos)?;
            if flag > 3
                || (flag > 2 && hd.version < 6)
                || (flag > 1 && hd.version < 4)
            {
                return Err(LercError::InvalidBlob);
            }
            let image_encode_mode = match flag {
                0 => ImageEncodeMode::Tiling,
                1 => ImageEncodeMode::DeltaHuffman,
                2 => ImageEncodeMode::Huffman,
                3 => ImageEncodeMode::DeltaDeltaHuffman,
                _ => return Err(LercError::InvalidBlob),
            };

            if image_encode_mode != ImageEncodeMode::Tiling {
                if hd.try_huffman_int() {
                    if image_encode_mode == ImageEncodeMode::DeltaHuffman
                        || (hd.version >= 4
                            && image_encode_mode == ImageEncodeMode::Huffman)
                    {
                        decode_huffman(
                            blob,
                            &mut pos,
                            &mut data,
                            &hd,
                            &mask,
                            image_encode_mode,
                        )?;
                    } else {
                        return Err(LercError::InvalidBlob);
                    }
                } else {
                    // try_huffman_flt: DeltaDeltaHuffman / lossless float
                    if image_encode_mode != ImageEncodeMode::DeltaDeltaHuffman {
                        return Err(LercError::InvalidBlob);
                    }
                    T::decode_lossless_flt(
                        blob,
                        &mut pos,
                        &mut data,
                        hd.n_cols as usize,
                        hd.n_rows as usize,
                        hd.n_depth as usize,
                    )?;
                }
                // noData remapping after Huffman (v6+).
                if hd.version >= 6 && hd.b_pass_no_data_values && hd.n_depth > 1 {
                    remap_no_data(
                        &mut data,
                        &mask,
                        hd.n_rows,
                        hd.n_cols,
                        hd.n_depth,
                        T::cast_f64(hd.no_data_val),
                        T::cast_f64(hd.no_data_val_orig),
                    );
                }
                return Ok((data, mask, hd));
            }
        }

        // Tile-based decode.
        read_tiles(blob, &mut pos, &mut data, &hd, &mask, &z_max_vec)?;
    } else {
        // Raw "one sweep" decode.
        read_data_one_sweep(blob, &mut pos, &mut data, &hd, &mask)?;
    }

    // noData remapping (v6+, nDepth > 1).
    if hd.version >= 6 && hd.b_pass_no_data_values && hd.n_depth > 1 {
        remap_no_data(
            &mut data,
            &mask,
            hd.n_rows,
            hd.n_cols,
            hd.n_depth,
            T::cast_f64(hd.no_data_val),
            T::cast_f64(hd.no_data_val_orig),
        );
    }

    Ok((data, mask, hd))
}

// ── dispatch_read_min_max ─────────────────────────────────────────────────────

/// Dispatch ReadMinMaxRanges for the data type of `hd`.
fn dispatch_read_min_max(
    blob: &[u8],
    pos: &mut usize,
    hd: &HeaderInfo,
) -> Result<(Vec<f64>, Vec<f64>), LercError> {
    let nd = hd.n_depth as usize;
    match hd.dt {
        DataType::I8  => read_min_max_ranges::<i8>(blob, pos, nd),
        DataType::U8  => read_min_max_ranges::<u8>(blob, pos, nd),
        DataType::I16 => read_min_max_ranges::<i16>(blob, pos, nd),
        DataType::U16 => read_min_max_ranges::<u16>(blob, pos, nd),
        DataType::I32 => read_min_max_ranges::<i32>(blob, pos, nd),
        DataType::U32 => read_min_max_ranges::<u32>(blob, pos, nd),
        DataType::F32 => read_min_max_ranges::<f32>(blob, pos, nd),
        DataType::F64 => read_min_max_ranges::<f64>(blob, pos, nd),
    }
}

// ── DecodeBand (dispatch on DataType) ────────────────────────────────────────

enum BandResult {
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

fn decode_band_dispatch(
    src: &[u8],
    blob_start: usize,
    dt: DataType,
) -> Result<(BandResult, BitMask, HeaderInfo), LercError> {
    match dt {
        DataType::I8 => {
            let (d, m, h) = decode_band_impl::<i8>(src, blob_start)?;
            Ok((BandResult::I8(d), m, h))
        }
        DataType::U8 => {
            let (d, m, h) = decode_band_impl::<u8>(src, blob_start)?;
            Ok((BandResult::U8(d), m, h))
        }
        DataType::I16 => {
            let (d, m, h) = decode_band_impl::<i16>(src, blob_start)?;
            Ok((BandResult::I16(d), m, h))
        }
        DataType::U16 => {
            let (d, m, h) = decode_band_impl::<u16>(src, blob_start)?;
            Ok((BandResult::U16(d), m, h))
        }
        DataType::I32 => {
            let (d, m, h) = decode_band_impl::<i32>(src, blob_start)?;
            Ok((BandResult::I32(d), m, h))
        }
        DataType::U32 => {
            let (d, m, h) = decode_band_impl::<u32>(src, blob_start)?;
            Ok((BandResult::U32(d), m, h))
        }
        DataType::F32 => {
            let (d, m, h) = decode_band_impl::<f32>(src, blob_start)?;
            Ok((BandResult::F32(d), m, h))
        }
        DataType::F64 => {
            let (d, m, h) = decode_band_impl::<f64>(src, blob_start)?;
            Ok((BandResult::F64(d), m, h))
        }
    }
}

// ── mask_to_bytes ─────────────────────────────────────────────────────────────

fn mask_to_bytes(mask: &BitMask) -> Vec<u8> {
    (0..mask.n_pixels() as i32)
        .map(|k| u8::from(mask.is_valid(k)))
        .collect()
}

// ── Public API: get_lerc_info ─────────────────────────────────────────────────

/// Parse metadata from a LERC blob without decoding pixel data.
///
/// Supports multi-band blobs (concatenated single-band Lerc2 blobs).
/// Returns `Err(UnsupportedFeature("Lerc1"))` for old CntZImage blobs.
pub fn get_lerc_info(src: &[u8]) -> Result<LercInfo, LercError> {
    // Check for Lerc1 (CntZImage magic).
    if src.starts_with(b"CntZImage ") {
        return Err(LercError::UnsupportedFeature("Lerc1"));
    }

    let (first_hd, has_mask) = get_header_info(src, 0).ok_or(LercError::InvalidBlob)?;
    if first_hd.version < 1 {
        return Err(LercError::InvalidBlob);
    }

    let mut n_masks: i32 = if has_mask || first_hd.num_valid_pixel == 0 { 1 } else { 0 };
    let mut n_uses_no_data: i32 = if first_hd.b_pass_no_data_values { 1 } else { 0 };

    let mut total_blob_size = first_hd.blob_size as i64;
    let mut n_bands = 1i32;
    let mut z_min = first_hd.z_min;
    let mut z_max = first_hd.z_max;
    let mut max_z_error = first_hd.max_z_error;

    if total_blob_size > src.len() as i64 {
        return Err(LercError::TruncatedBlob);
    }

    let mut try_next = first_hd.version <= 5 || first_hd.n_blobs_more > 0;
    let mut prev_num_valid = first_hd.num_valid_pixel;

    while try_next {
        let off = total_blob_size as usize;
        let Some((hd, bm)) = get_header_info(src, off) else {
            break;
        };
        if hd.n_depth != first_hd.n_depth
            || hd.n_cols != first_hd.n_cols
            || hd.n_rows != first_hd.n_rows
            || hd.dt as i32 != first_hd.dt as i32
        {
            return Err(LercError::InvalidBlob);
        }

        try_next = hd.version <= 5 || hd.n_blobs_more > 0;

        if hd.b_pass_no_data_values {
            n_uses_no_data += 1;
        }
        if bm || hd.num_valid_pixel != prev_num_valid {
            n_masks = 2;
        }
        prev_num_valid = hd.num_valid_pixel;

        if total_blob_size > i64::MAX - hd.blob_size as i64 {
            return Err(LercError::InvalidBlob);
        }
        total_blob_size += hd.blob_size as i64;
        if total_blob_size > src.len() as i64 {
            return Err(LercError::TruncatedBlob);
        }

        z_min = z_min.min(hd.z_min);
        z_max = z_max.max(hd.z_max);
        max_z_error = max_z_error.max(hd.max_z_error);
        n_bands += 1;
    }

    let final_n_masks = if n_masks > 1 { n_bands } else { n_masks };
    let final_n_uses_no_data = if n_uses_no_data > 0 { n_bands } else { 0 };

    Ok(LercInfo {
        version: first_hd.version,
        n_depth: first_hd.n_depth,
        n_cols: first_hd.n_cols,
        n_rows: first_hd.n_rows,
        num_valid_pixel: first_hd.num_valid_pixel,
        n_bands,
        blob_size: total_blob_size as i32,
        n_masks: final_n_masks,
        n_uses_no_data_value: final_n_uses_no_data,
        data_type: first_hd.dt,
        z_min,
        z_max,
        max_z_error,
    })
}

// ── Public API: decode ────────────────────────────────────────────────────────

/// Decode a LERC blob into pixel data, validity mask, and metadata.
///
/// The returned `DecodedData::data` layout is:
/// `[band * n_rows * n_cols * n_depth + row * n_cols * n_depth + col * n_depth + depth]`
///
/// `valid_pixels` is `None` if all pixels in every band are valid, or
/// `Some(vec)` with length `n_masks * n_rows * n_cols` (1 = valid, 0 = invalid).
pub fn decode(src: &[u8]) -> Result<DecodedData, LercError> {
    let info = get_lerc_info(src)?;

    let n_pix = info.n_rows as usize * info.n_cols as usize;
    let n_elem = n_pix * info.n_depth as usize;
    let n_bands = info.n_bands as usize;

    // Decode each band.
    let mut blob_offset: usize = 0;
    let mut band_data: Vec<BandResult> = Vec::with_capacity(n_bands);
    let mut band_masks: Vec<Vec<u8>> = Vec::with_capacity(n_bands);
    let mut no_data_values: Vec<f64> = vec![0.0; n_bands];
    let mut any_no_data = false;

    for _ib in 0..n_bands {
        let (band, mask, hd) = decode_band_dispatch(src, blob_offset, info.data_type)?;
        blob_offset += hd.blob_size as usize;

        band_masks.push(mask_to_bytes(&mask));

        if hd.b_pass_no_data_values {
            no_data_values[_ib] = hd.no_data_val_orig;
            any_no_data = true;
        }

        band_data.push(band);
    }

    // Merge per-band pixel data into one flat array.
    let data = merge_band_data(band_data, n_bands, n_elem)?;

    // Determine valid_pixels.
    let all_valid = band_masks
        .iter()
        .all(|m| m.iter().all(|&b| b == 1));

    let valid_pixels = if all_valid {
        None
    } else if info.n_masks == 1 {
        // All bands share the same mask (just return band 0's mask).
        Some(band_masks.remove(0))
    } else {
        // Per-band masks: concatenate.
        let mut combined = Vec::with_capacity(info.n_masks as usize * n_pix);
        for m in band_masks.into_iter().take(info.n_masks as usize) {
            combined.extend_from_slice(&m);
        }
        Some(combined)
    };

    let no_data_out = if any_no_data {
        Some(no_data_values)
    } else {
        None
    };

    Ok(DecodedData {
        data,
        valid_pixels,
        no_data_values: no_data_out,
        info,
    })
}

// ── merge_band_data ───────────────────────────────────────────────────────────

fn merge_band_data(
    bands: Vec<BandResult>,
    n_bands: usize,
    n_elem_per_band: usize,
) -> Result<LercData, LercError> {
    if bands.is_empty() {
        return Err(LercError::InvalidBlob);
    }

    macro_rules! merge {
        ($variant:ident, $inner:ty) => {{
            let mut out: Vec<$inner> = Vec::with_capacity(n_bands * n_elem_per_band);
            for band in bands {
                match band {
                    BandResult::$variant(v) => out.extend_from_slice(&v),
                    _ => return Err(LercError::InvalidBlob),
                }
            }
            LercData::$variant(out)
        }};
    }

    let data = match &bands[0] {
        BandResult::I8(_)  => merge!(I8,  i8),
        BandResult::U8(_)  => merge!(U8,  u8),
        BandResult::I16(_) => merge!(I16, i16),
        BandResult::U16(_) => merge!(U16, u16),
        BandResult::I32(_) => merge!(I32, i32),
        BandResult::U32(_) => merge!(U32, u32),
        BandResult::F32(_) => merge!(F32, f32),
        BandResult::F64(_) => merge!(F64, f64),
    };
    Ok(data)
}
