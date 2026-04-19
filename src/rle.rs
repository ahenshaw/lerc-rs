use crate::error::LercError;

fn read_i16_le(data: &[u8], pos: usize) -> Option<i16> {
    if pos + 2 > data.len() {
        return None;
    }
    Some(i16::from_le_bytes([data[pos], data[pos + 1]]))
}

/// Decompress RLE-encoded mask bytes from `src` into `dst`.
///
/// Encoding:
/// - A positive count `n` means: copy the following `n` literal bytes.
/// - A negative count `-n` means: repeat the following 1 byte `n` times.
/// - `i16::MIN` (-32768) is the end-of-stream terminator.
pub(crate) fn rle_decompress(src: &[u8], dst: &mut [u8]) -> Result<(), LercError> {
    let mut sp = 0usize;
    let mut dp = 0usize;

    loop {
        let cnt = read_i16_le(src, sp).ok_or(LercError::TruncatedBlob)?;
        sp += 2;

        if cnt == i16::MIN {
            // end-of-stream terminator
            break;
        }

        if cnt > 0 {
            let n = cnt as usize;
            if sp + n > src.len() {
                return Err(LercError::TruncatedBlob);
            }
            if dp + n > dst.len() {
                return Err(LercError::DecodeFailed);
            }
            dst[dp..dp + n].copy_from_slice(&src[sp..sp + n]);
            sp += n;
            dp += n;
        } else {
            // cnt < 0 (and != i16::MIN)
            let n = (-(cnt as i32)) as usize;
            if sp >= src.len() {
                return Err(LercError::TruncatedBlob);
            }
            if dp + n > dst.len() {
                return Err(LercError::DecodeFailed);
            }
            let byte = src[sp];
            sp += 1;
            dst[dp..dp + n].fill(byte);
            dp += n;
        }
    }

    Ok(())
}
