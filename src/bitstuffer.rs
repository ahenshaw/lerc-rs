/// Decode a LERC2 BitStuffer2-encoded unsigned integer array.
///
/// Reads from `src[*pos..]` and advances `*pos` by the consumed bytes.
/// `max_count` is a safety bound on the number of elements.
/// `version` is the Lerc2 file version (affects bit-packing scheme).
use crate::error::LercError;

/// Number of 4-byte "tail" words that don't need to be present in the stream
/// because their bits are padding zeros.
fn num_tail_bytes_not_needed(num_elem: usize, num_bits: usize) -> usize {
    let tail_bits = (num_elem * num_bits) & 31;
    let tail_bytes = (tail_bits + 7) >> 3;
    if tail_bytes > 0 { 4 - tail_bytes } else { 0 }
}

fn decode_uint(src: &[u8], pos: &mut usize, nb: usize) -> Option<u32> {
    if *pos + nb > src.len() {
        return None;
    }
    let val = match nb {
        1 => src[*pos] as u32,
        2 => u16::from_le_bytes([src[*pos], src[*pos + 1]]) as u32,
        4 => u32::from_le_bytes([src[*pos], src[*pos + 1], src[*pos + 2], src[*pos + 3]]),
        _ => return None,
    };
    *pos += nb;
    Some(val)
}

/// BitUnStuff v3+ (Lerc2 v3+): bits packed LSB-first within little-endian uint32 words.
///
/// Writes decoded values into `out` (clearing it first and reusing its allocation).
/// Returns false if the source is truncated.
fn bit_unstuff_v3(
    src: &[u8],
    pos: &mut usize,
    num_elem: usize,
    num_bits: usize,
    out: &mut Vec<u32>,
) -> bool {
    let total_bits = num_elem as u64 * num_bits as u64;
    let num_uints = ((total_bits + 31) / 32) as usize;
    let num_bytes = num_uints * 4;
    let ntbnn = num_tail_bytes_not_needed(num_elem, num_bits);
    let num_bytes_used = num_bytes.saturating_sub(ntbnn);

    if *pos + num_bytes_used > src.len() {
        return false;
    }

    let raw = &src[*pos..*pos + num_bytes_used];
    *pos += num_bytes_used;

    // Build an iterator of LE u32 words directly from the raw bytes.
    // The last partial chunk is automatically zero-padded (correct for LSB-first).
    let mut words = raw.chunks(4).map(|c| {
        let mut w = [0u8; 4];
        w[..c.len()].copy_from_slice(c);
        u32::from_le_bytes(w)
    });

    // Extract num_bits-wide values using a u64 accumulator.
    let mask: u32 = if num_bits < 32 { (1u32 << num_bits) - 1 } else { u32::MAX };

    out.clear();
    out.reserve(num_elem);

    let mut acc = 0u64;
    let mut bits_in_acc: u32 = 0;

    for _ in 0..num_elem {
        if bits_in_acc < num_bits as u32 {
            let Some(w) = words.next() else { return false; };
            acc |= (w as u64) << bits_in_acc;
            bits_in_acc += 32;
        }
        out.push(acc as u32 & mask);
        acc >>= num_bits;
        bits_in_acc -= num_bits as u32;
    }

    true
}

/// BitUnStuff pre-v3 (Lerc2 v1/v2): bits packed MSB-first within LE uint32 words.
fn bit_unstuff_pre_v3(
    src: &[u8],
    pos: &mut usize,
    num_elem: usize,
    num_bits: usize,
    out: &mut Vec<u32>,
) -> bool {
    let total_bits = num_elem as u64 * num_bits as u64;
    let num_uints = ((total_bits + 31) / 32) as usize;
    let ntbnn = num_tail_bytes_not_needed(num_elem, num_bits);
    let n_bytes_to_copy = (num_elem * num_bits + 7) / 8;

    if *pos + n_bytes_to_copy > src.len() {
        return false;
    }

    // Read bytes as LE uint32 words.
    let mut buf = vec![0u32; num_uints];
    for i in 0..n_bytes_to_copy {
        buf[i / 4] |= (src[*pos + i] as u32) << ((i % 4) * 8);
    }
    *pos += n_bytes_to_copy;

    // Apply the last-uint shift: moves data into the MSB positions so that
    // the MSB-first read below works correctly.
    if ntbnn > 0 && num_uints > 0 {
        for _ in 0..ntbnn {
            buf[num_uints - 1] <<= 8;
        }
    }

    // Unpack: MSB-first within each uint32.
    out.clear();
    out.reserve(num_elem);

    let mut si = 0usize;
    let mut bit_pos = 0i32;

    for _ in 0..num_elem {
        let val = if 32 - bit_pos >= num_bits as i32 {
            let n = buf[si].wrapping_shl(bit_pos as u32);
            let v = n.wrapping_shr((32 - num_bits) as u32);
            bit_pos += num_bits as i32;
            if bit_pos == 32 {
                bit_pos = 0;
                si += 1;
            }
            v
        } else {
            // Span two words.
            let n = buf[si].wrapping_shl(bit_pos as u32);
            let mut v = n.wrapping_shr((32 - num_bits) as u32);
            bit_pos -= 32 - num_bits as i32;
            si += 1;
            v |= buf[si].wrapping_shr((32 - bit_pos) as u32);
            v
        };
        out.push(val);
    }

    true
}

/// Decode a BitStuffer2 block, writing results into `out`.
///
/// `out` is cleared and refilled; its existing allocation is reused when
/// capacity is sufficient, avoiding per-tile heap allocation.
pub(crate) fn decode(
    src: &[u8],
    pos: &mut usize,
    max_count: usize,
    version: i32,
    out: &mut Vec<u32>,
) -> Result<(), LercError> {
    if *pos >= src.len() {
        return Err(LercError::TruncatedBlob);
    }
    let hdr = src[*pos];
    *pos += 1;

    let bits67 = (hdr >> 6) as usize;
    // nb = number of bytes used to encode the element count
    let nb = if bits67 == 0 { 4 } else { 3 - bits67 };
    let do_lut = (hdr & (1 << 5)) != 0;
    let num_bits = (hdr & 31) as usize;

    let num_elements =
        decode_uint(src, pos, nb).ok_or(LercError::TruncatedBlob)? as usize;
    if num_elements > max_count {
        return Err(LercError::DecodeFailed);
    }

    if !do_lut {
        if num_bits == 0 {
            // All values are zero (quantized min).
            out.clear();
            out.resize(num_elements, 0);
            return Ok(());
        }
        let ok = if version >= 3 {
            bit_unstuff_v3(src, pos, num_elements, num_bits, out)
        } else {
            bit_unstuff_pre_v3(src, pos, num_elements, num_bits, out)
        };
        ok.then_some(()).ok_or(LercError::TruncatedBlob)
    } else {
        // LUT mode: decode lut values into a temporary buffer, then indices into `out`.
        if num_bits == 0 {
            return Err(LercError::DecodeFailed);
        }
        if *pos >= src.len() {
            return Err(LercError::TruncatedBlob);
        }
        let n_lut_byte = src[*pos] as usize;
        *pos += 1;
        let n_lut = n_lut_byte - 1; // size of LUT excluding implicit 0

        // Decode LUT values into a small temporary buffer.
        let mut lut = Vec::new();
        let ok = if version >= 3 {
            bit_unstuff_v3(src, pos, n_lut, num_bits, &mut lut)
        } else {
            bit_unstuff_pre_v3(src, pos, n_lut, num_bits, &mut lut)
        };
        if !ok {
            return Err(LercError::TruncatedBlob);
        }

        // nBitsLut = ceil(log2(n_lut + 1))
        let mut n_bits_lut = 0usize;
        let mut tmp = n_lut;
        while tmp > 0 {
            n_bits_lut += 1;
            tmp >>= 1;
        }
        if n_bits_lut == 0 {
            return Err(LercError::DecodeFailed);
        }

        // Decode indices into `out`.
        let ok = if version >= 3 {
            bit_unstuff_v3(src, pos, num_elements, n_bits_lut, out)
        } else {
            bit_unstuff_pre_v3(src, pos, num_elements, n_bits_lut, out)
        };
        if !ok {
            return Err(LercError::TruncatedBlob);
        }

        // Prepend the implicit 0 entry and map indices → values.
        lut.insert(0, 0);
        for idx in out.iter_mut() {
            let i = *idx as usize;
            if i >= lut.len() {
                return Err(LercError::DecodeFailed);
            }
            *idx = lut[i];
        }

        Ok(())
    }
}
