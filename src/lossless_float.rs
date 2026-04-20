/// Lossless float decoder for LERC2 v6 DeltaDeltaHuffman encoding.
///
/// Implements the `DecodeHuffmanFltSlice` logic from `fpl_Lerc2Ext.cpp`.
/// Each byte plane is compressed independently using `fpl_EsriHuffman`,
/// then an inverse predictor and (for f32) a bit rearrangement are undone.
use crate::{error::LercError, huffman::HuffmanDecoder};

const MAX_DELTA: usize = 5;

// ── Arithmetic helpers ────────────────────────────────────────────────────────

/// ADD32_BIT_FLT: adds the "bit-rearranged" f32 representation produced by
/// `moveBits2Front` with field-independent wrapping:
///   - low 23 bits (mantissa): wraps at 2^23
///   - high 9 bits (exponent 8 bits + sign 1 bit): wraps at 2^9
#[inline]
fn add32_bit_flt(a: u32, b: u32) -> u32 {
    let mantissa = a.wrapping_add(b) & 0x007F_FFFF;
    let ae = (a >> 23) & 0x1FF;
    let be = (b >> 23) & 0x1FF;
    mantissa | ((ae.wrapping_add(be) & 0x1FF) << 23)
}

/// ADD64_BIT_DBL: adds raw f64 bit patterns with field-independent wrapping:
///   - low 52 bits (mantissa): wraps at 2^52
///   - high 12 bits (exponent 11 bits + sign 1 bit): wraps at 2^12
#[inline]
fn add64_bit_dbl(a: u64, b: u64) -> u64 {
    let mantissa = a.wrapping_add(b) & 0x000F_FFFF_FFFF_FFFF;
    let ae = (a >> 52) & 0xFFF;
    let be = (b >> 52) & 0xFFF;
    mantissa | ((ae.wrapping_add(be) & 0xFFF) << 52)
}

/// Inverse of `moveBits2Front`: restores the standard IEEE 754 f32 bit layout.
///
/// Input after moveBits2Front: [exp:31:24][sign:23][mantissa:22:0]
/// Output (standard f32):      [sign:31][exp:30:23][mantissa:22:0]
#[inline]
fn undo_move_bits2front(a: u32) -> u32 {
    let mantissa = a & 0x007F_FFFF;
    let exponent = (a >> 24) & 0xFF;
    let sign = (a >> 23) & 0x01;
    mantissa | (exponent << 23) | (sign << 31)
}

// ── Byte-delta restore ────────────────────────────────────────────────────────

/// Inverse of `setDerivative`: cumulative sum of bytes applied `level` times.
fn restore_sequence(data: &mut [u8], level: usize) {
    // Level applied from highest to 1 (matching the C++ loop direction).
    for l in (1..=level).rev() {
        for i in l..data.len() {
            data[i] = data[i].wrapping_add(data[i - 1]);
        }
    }
}

// ── Predictor inverse: DELTA1 (restoreBlockSequence) ─────────────────────────

fn restore_block_sequence_f32(delta: usize, pixels: &mut [u32], n_cols: usize) {
    if delta == 0 {
        return;
    }
    if delta >= 2 {
        for row in pixels.chunks_mut(n_cols) {
            for i in 2..row.len() {
                let prev = row[i - 1];
                row[i] = add32_bit_flt(row[i], prev);
            }
        }
    }
    // First-order row-wise prefix sum (always when delta > 0).
    for row in pixels.chunks_mut(n_cols) {
        for i in 1..row.len() {
            let prev = row[i - 1];
            row[i] = add32_bit_flt(row[i], prev);
        }
    }
}

fn restore_block_sequence_f64(delta: usize, pixels: &mut [u64], n_cols: usize) {
    if delta == 0 {
        return;
    }
    if delta >= 2 {
        for row in pixels.chunks_mut(n_cols) {
            for i in 2..row.len() {
                let prev = row[i - 1];
                row[i] = add64_bit_dbl(row[i], prev);
            }
        }
    }
    for row in pixels.chunks_mut(n_cols) {
        for i in 1..row.len() {
            let prev = row[i - 1];
            row[i] = add64_bit_dbl(row[i], prev);
        }
    }
}

// ── Predictor inverse: ROWS_COLS (restoreCrossBytes) ─────────────────────────

fn restore_cross_bytes_f32(delta: usize, pixels: &mut [u32], n_cols: usize, n_rows: usize) {
    if delta == 0 {
        return;
    }
    if delta >= 2 {
        // Column-wise cumsum.
        for col in 0..n_cols {
            for row in 1..n_rows {
                let prev = pixels[(row - 1) * n_cols + col];
                let cur = pixels[row * n_cols + col];
                pixels[row * n_cols + col] = add32_bit_flt(cur, prev);
            }
        }
    }
    // Row-wise cumsum (always for ROWS_COLS).
    for row in 0..n_rows {
        for col in 1..n_cols {
            let prev = pixels[row * n_cols + col - 1];
            let cur = pixels[row * n_cols + col];
            pixels[row * n_cols + col] = add32_bit_flt(cur, prev);
        }
    }
}

fn restore_cross_bytes_f64(delta: usize, pixels: &mut [u64], n_cols: usize, n_rows: usize) {
    if delta == 0 {
        return;
    }
    if delta >= 2 {
        for col in 0..n_cols {
            for row in 1..n_rows {
                let prev = pixels[(row - 1) * n_cols + col];
                let cur = pixels[row * n_cols + col];
                pixels[row * n_cols + col] = add64_bit_dbl(cur, prev);
            }
        }
    }
    for row in 0..n_rows {
        for col in 1..n_cols {
            let prev = pixels[row * n_cols + col - 1];
            let cur = pixels[row * n_cols + col];
            pixels[row * n_cols + col] = add64_bit_dbl(cur, prev);
        }
    }
}

// ── PackBits RLE decoder ──────────────────────────────────────────────────────

/// Decompress PackBits into `out` (appending).  `out` must have been `clear()`ed
/// and `reserve()`d by the caller.
fn decode_packbits_into(data: &[u8], expected: usize, out: &mut Vec<u8>) -> Result<(), LercError> {
    let mut i = 0;
    while i < data.len() {
        let b = data[i];
        i += 1;
        if b <= 127 {
            // Literal run: copy b+1 bytes.
            let count = b as usize + 1;
            if i + count > data.len() {
                return Err(LercError::TruncatedBlob);
            }
            out.extend_from_slice(&data[i..i + count]);
            i += count;
        } else {
            // Repeat run: b-126 copies of the next byte.
            let count = b as usize - 126;
            if i >= data.len() {
                return Err(LercError::TruncatedBlob);
            }
            let byte = data[i];
            out.resize(out.len() + count, byte);
            i += 1;
        }
    }
    if out.len() != expected {
        return Err(LercError::DecodeFailed);
    }
    Ok(())
}

// ── Byte-plane decompression (fpl_EsriHuffman::DecodeHuffman) ────────────────

/// Decompress one byte plane into `out`, reusing its allocation.
///
/// `compressed` contains the mode byte followed by the compressed payload.
/// `out` is cleared and filled with exactly `expected_size` bytes.
fn decode_byte_plane_into(
    compressed: &[u8],
    expected_size: usize,
    out: &mut Vec<u8>,
) -> Result<(), LercError> {
    out.clear();
    if compressed.is_empty() {
        return Err(LercError::TruncatedBlob);
    }
    match compressed[0] {
        0 => {
            // HUFFMAN_NORMAL: standard LERC2 Huffman, version 5.
            let mut pos = 1usize; // skip mode byte
            let huff = HuffmanDecoder::from_blob(compressed, &mut pos, 5)?;
            out.reserve(expected_size);
            let mut bit_pos = 0i32;
            for _ in 0..expected_size {
                let sym = if let Some(s) = huff.decode_one_fast(compressed, &mut pos, &mut bit_pos)
                {
                    s
                } else {
                    huff.decode_one(compressed, &mut pos, &mut bit_pos)?
                };
                out.push(sym as u8);
            }
        }
        1 => {
            // HUFFMAN_RLE: single byte value repeated N times.
            // Format: [1][value][count:u32le]
            if compressed.len() < 6 {
                return Err(LercError::TruncatedBlob);
            }
            let value = compressed[1];
            let count = u32::from_le_bytes(compressed[2..6].try_into().unwrap()) as usize;
            if count != expected_size {
                return Err(LercError::InvalidBlob);
            }
            out.resize(expected_size, value);
        }
        2 => {
            // HUFFMAN_NO_ENCODING: raw bytes follow the mode byte.
            if compressed.len() < 1 + expected_size {
                return Err(LercError::TruncatedBlob);
            }
            out.extend_from_slice(&compressed[1..1 + expected_size]);
        }
        3 => {
            // HUFFMAN_PACKBITS: PackBits RLE after the mode byte.
            out.reserve(expected_size);
            decode_packbits_into(&compressed[1..], expected_size, out)?;
        }
        _ => return Err(LercError::InvalidBlob),
    }
    if out.len() != expected_size {
        return Err(LercError::DecodeFailed);
    }
    Ok(())
}

// ── Public entry points ───────────────────────────────────────────────────────

pub(crate) fn decode_lossless_f32(
    src: &[u8],
    pos: &mut usize,
    data: &mut [f32],
    n_cols: usize,
    n_rows: usize,
    n_depth: usize,
) -> Result<(), LercError> {
    // C++ flips the dimensions when n_depth > 1 (each "row" = one pixel's depth values).
    let (iw, ih) = if n_depth == 1 {
        (n_cols, n_rows)
    } else {
        (n_depth, n_cols * n_rows)
    };
    let expected_size = iw * ih;

    if *pos >= src.len() {
        return Err(LercError::TruncatedBlob);
    }
    let pred_code = src[*pos];
    *pos += 1;
    if pred_code > 2 {
        return Err(LercError::InvalidBlob);
    }

    // One integer array for the assembled pixel data; one scratch buffer
    // reused across all 4 planes instead of allocating a separate Vec per plane.
    let mut pixels = vec![0u32; expected_size];
    let mut scratch: Vec<u8> = Vec::with_capacity(expected_size);

    for _ in 0..4usize {
        if *pos + 6 > src.len() {
            return Err(LercError::TruncatedBlob);
        }
        let byte_index = src[*pos] as usize;
        *pos += 1;
        let best_level = src[*pos] as usize;
        *pos += 1;
        let compressed_size = u32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap()) as usize;
        *pos += 4;

        if byte_index >= 4 {
            return Err(LercError::InvalidBlob);
        }
        if best_level > MAX_DELTA {
            return Err(LercError::InvalidBlob);
        }
        if *pos + compressed_size > src.len() {
            return Err(LercError::TruncatedBlob);
        }

        let compressed = &src[*pos..*pos + compressed_size];
        *pos += compressed_size;

        // Decompress into the reused scratch buffer.
        decode_byte_plane_into(compressed, expected_size, &mut scratch)?;
        restore_sequence(&mut scratch, best_level);

        // Accumulate this plane's bytes into pixels at the correct bit position.
        // Both arrays are accessed sequentially — cache-friendly.
        let shift = byte_index as u32 * 8;
        for (px, &b) in pixels.iter_mut().zip(scratch.iter()) {
            *px |= (b as u32) << shift;
        }
    }

    // Apply inverse predictor.
    match pred_code {
        0 => {}
        1 => restore_block_sequence_f32(1, &mut pixels, iw),
        2 => restore_cross_bytes_f32(2, &mut pixels, iw, ih),
        _ => return Err(LercError::InvalidBlob),
    }

    // Undo moveBits2Front and write directly into the output slice —
    // no intermediate byte buffer needed.
    for (px, out) in pixels.iter().zip(data.iter_mut()) {
        *out = f32::from_bits(undo_move_bits2front(*px));
    }

    Ok(())
}

pub(crate) fn decode_lossless_f64(
    src: &[u8],
    pos: &mut usize,
    data: &mut [f64],
    n_cols: usize,
    n_rows: usize,
    n_depth: usize,
) -> Result<(), LercError> {
    let (iw, ih) = if n_depth == 1 {
        (n_cols, n_rows)
    } else {
        (n_depth, n_cols * n_rows)
    };
    let expected_size = iw * ih;

    if *pos >= src.len() {
        return Err(LercError::TruncatedBlob);
    }
    let pred_code = src[*pos];
    *pos += 1;
    if pred_code > 2 {
        return Err(LercError::InvalidBlob);
    }

    let mut pixels = vec![0u64; expected_size];
    let mut scratch: Vec<u8> = Vec::with_capacity(expected_size);

    for _ in 0..8usize {
        if *pos + 6 > src.len() {
            return Err(LercError::TruncatedBlob);
        }
        let byte_index = src[*pos] as usize;
        *pos += 1;
        let best_level = src[*pos] as usize;
        *pos += 1;
        let compressed_size = u32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap()) as usize;
        *pos += 4;

        if byte_index >= 8 {
            return Err(LercError::InvalidBlob);
        }
        if best_level > MAX_DELTA {
            return Err(LercError::InvalidBlob);
        }
        if *pos + compressed_size > src.len() {
            return Err(LercError::TruncatedBlob);
        }

        let compressed = &src[*pos..*pos + compressed_size];
        *pos += compressed_size;

        decode_byte_plane_into(compressed, expected_size, &mut scratch)?;
        restore_sequence(&mut scratch, best_level);

        let shift = byte_index as u32 * 8;
        for (px, &b) in pixels.iter_mut().zip(scratch.iter()) {
            *px |= (b as u64) << shift;
        }
    }

    match pred_code {
        0 => {}
        1 => restore_block_sequence_f64(1, &mut pixels, iw),
        2 => restore_cross_bytes_f64(2, &mut pixels, iw, ih),
        _ => return Err(LercError::InvalidBlob),
    }

    // No float bit transform for f64; write directly to output.
    for (px, out) in pixels.iter().zip(data.iter_mut()) {
        *out = f64::from_bits(*px);
    }

    Ok(())
}
