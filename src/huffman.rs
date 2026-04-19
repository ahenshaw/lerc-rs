/// Huffman decoder for LERC2 Huffman-coded image data.
///
/// Code tables use canonical Huffman codes stored MSB-first in packed uint32 words.
/// A 12-bit LUT covers the common case; a binary tree handles codes > 12 bits.
use crate::{bitstuffer, error::LercError};

const MAX_BITS_LUT: usize = 12;

/// One node in the decode tree for codes longer than 12 bits.
#[derive(Clone, Default)]
struct TreeNode {
    /// Index into `tree` for child 0/1 (positive), or special sentinel:
    /// -1 = absent; ≤ -2 = leaf with symbol = -(child + 2).
    child: [i32; 2],
}

/// Huffman decoder built from a ReadCodeTable blob.
pub(crate) struct HuffmanDecoder {
    /// LUT indexed by the top `MAX_BITS_LUT` bits of the bit stream.
    /// Entry: (code_length_in_bits, symbol).  code_length == 0 means "not in LUT".
    lut: Vec<(u8, u16)>,
    /// Decode tree for codes with length > MAX_BITS_LUT (rare).
    tree: Vec<TreeNode>,
    has_tree: bool,
    num_bits_to_skip: i32,
}

impl HuffmanDecoder {
    /// Parse a Huffman code table from the blob at `*pos` and build the decoder.
    pub fn from_blob(
        src: &[u8],
        pos: &mut usize,
        lerc2_version: i32,
    ) -> Result<Self, LercError> {
        if *pos + 16 > src.len() {
            return Err(LercError::TruncatedBlob);
        }
        let huffman_version =
            i32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        let table_size =
            i32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap()) as usize;
        *pos += 4;
        let i0 = i32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        let i1 = i32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;

        if huffman_version < 2 || i0 >= i1 || i0 < 0 || table_size == 0
            || table_size > (1 << 15)
        {
            return Err(LercError::InvalidBlob);
        }

        fn wrap_idx(i: i32, size: usize) -> usize {
            let s = size as i32;
            (i - if i < s { 0 } else { s }) as usize
        }

        let code_range = (i1 - i0) as usize;

        // Decode the bit-stuffed code lengths.
        let mut code_lengths = Vec::new();
        bitstuffer::decode(src, pos, code_range, lerc2_version, &mut code_lengths)?;
        if code_lengths.len() != code_range {
            return Err(LercError::InvalidBlob);
        }

        let mut code_table = vec![(0u16, 0u32); table_size];
        for (offset, &len) in code_lengths.iter().enumerate() {
            let k = wrap_idx(i0 + offset as i32, table_size);
            code_table[k].0 = len as u16;
        }

        // Decode the bit-stuffed canonical codes (MSB-first uint32 stream).
        // This mirrors BitUnStuffCodes in the C++ implementation.
        let ptr0 = *pos;
        let mut bit_pos = 0i32;
        for i in i0..i1 {
            let k = wrap_idx(i, table_size);
            let len = code_table[k].0 as i32;
            if len > 0 {
                if *pos + 4 > src.len() {
                    return Err(LercError::TruncatedBlob);
                }
                let temp =
                    u32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
                let code = (temp << bit_pos as u32) >> (32 - len as u32);

                if 32 - bit_pos >= len {
                    bit_pos += len;
                    if bit_pos == 32 {
                        bit_pos = 0;
                        *pos += 4;
                    }
                } else {
                    bit_pos += len - 32;
                    *pos += 4;
                    if *pos + 4 > src.len() {
                        return Err(LercError::TruncatedBlob);
                    }
                    let temp2 =
                        u32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
                    code_table[k].1 = code | (temp2 >> (32 - bit_pos as u32));
                    continue;
                }
                code_table[k].1 = code;
            }
        }
        // Advance past the current partial uint32 (and the real check in C++).
        let consumed = *pos - ptr0;
        let total = consumed + if bit_pos > 0 { 4 } else { 0 };
        *pos = ptr0 + total;

        // Build decode LUT and optional tree.
        let lut_size = 1usize << MAX_BITS_LUT;
        let mut lut = vec![(0u8, 0u16); lut_size];
        let mut need_tree = false;
        let mut min_zero_bits = 32i32;

        for i in i0..i1 {
            let k = wrap_idx(i, table_size);
            let len = code_table[k].0 as i32;
            if len == 0 {
                continue;
            }
            let code = code_table[k].1;

            if len <= MAX_BITS_LUT as i32 {
                let shift = (MAX_BITS_LUT as i32 - len) as u32;
                let base = (code << shift) as usize;
                let count = 1usize << shift;
                for j in 0..count {
                    lut[base | j] = (len as u8, k as u16);
                }
            } else {
                need_tree = true;
                // Count leading zero bits in the canonical code
                // (the number of zero bits before the first 1-bit when
                //  viewed as a `len`-bit quantity).
                let shift = if code == 0 {
                    0i32
                } else {
                    32i32 - code.leading_zeros() as i32
                };
                let zero_bits = len - shift;
                if zero_bits < min_zero_bits {
                    min_zero_bits = zero_bits;
                }
            }
        }

        let num_bits_to_skip = if need_tree { min_zero_bits.max(0) } else { 0 };
        let mut tree: Vec<TreeNode> = Vec::new();

        if need_tree {
            // Allocate root (index 0).
            tree.push(TreeNode {
                child: [-1, -1],
            });

            for i in i0..i1 {
                let k = wrap_idx(i, table_size);
                let len = code_table[k].0 as i32;
                if len <= MAX_BITS_LUT as i32 || len == 0 {
                    continue;
                }
                let code = code_table[k].1;
                let effective_len = len - num_bits_to_skip;

                let mut node_idx = 0usize;
                let mut j = effective_len - 1;
                loop {
                    let bit = ((code >> j as u32) & 1) as usize;
                    if j == 0 {
                        // Leaf: encode as -(k+2) so k=0 gives -2, not -1 (the "absent" sentinel)
                        tree[node_idx].child[bit] = -(k as i32) - 2;
                        break;
                    }
                    let child = tree[node_idx].child[bit];
                    if child < 0 {
                        // Create new internal node.
                        let new_idx = tree.len() as i32;
                        tree.push(TreeNode {
                            child: [-1, -1],
                        });
                        tree[node_idx].child[bit] = new_idx;
                        node_idx = new_idx as usize;
                    } else {
                        node_idx = child as usize;
                    }
                    j -= 1;
                }
            }
        }

        Ok(Self {
            lut,
            tree,
            has_tree: need_tree,
            num_bits_to_skip,
        })
    }

    /// Decode one symbol from the MSB-first uint32 bit stream.
    ///
    /// `pos` points to the start of the current uint32; `bit_pos` is the
    /// bit offset within that uint32 (0 = MSB).  Both are updated on exit.
    pub fn decode_one(
        &self,
        src: &[u8],
        pos: &mut usize,
        bit_pos: &mut i32,
    ) -> Result<i32, LercError> {
        if *pos + 4 > src.len() {
            return Err(LercError::TruncatedBlob);
        }
        let temp = u32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());

        // Peek at the next MAX_BITS_LUT bits.
        let lut_bits = MAX_BITS_LUT as i32;
        let val_tmp = if 32 - *bit_pos >= lut_bits {
            (temp << *bit_pos as u32) >> (32 - lut_bits as u32)
        } else {
            if *pos + 8 > src.len() {
                return Err(LercError::TruncatedBlob);
            }
            let temp2 =
                u32::from_le_bytes(src[*pos + 4..*pos + 8].try_into().unwrap());
            let hi = (temp << *bit_pos as u32) >> (32 - lut_bits as u32);
            let lo = temp2 >> (64i32 - lut_bits - *bit_pos) as u32;
            hi | lo
        } as usize;

        let (code_len, symbol) = self.lut[val_tmp];

        if code_len > 0 {
            *bit_pos += code_len as i32;
            if *bit_pos >= 32 {
                *bit_pos -= 32;
                *pos += 4;
            }
            return Ok(symbol as i32);
        }

        // Fallback: tree traversal for codes > MAX_BITS_LUT bits.
        if !self.has_tree {
            return Err(LercError::DecodeFailed);
        }

        // Skip the leading zero bits shared by all long codes.
        *bit_pos += self.num_bits_to_skip;
        if *bit_pos >= 32 {
            *bit_pos -= 32;
            *pos += 4;
        }

        let mut node_idx = 0i32;
        loop {
            if *pos + 4 > src.len() {
                return Err(LercError::TruncatedBlob);
            }
            let t = u32::from_le_bytes(src[*pos..*pos + 4].try_into().unwrap());
            let bit = ((t << *bit_pos as u32) >> 31) as usize;
            *bit_pos += 1;
            if *bit_pos == 32 {
                *bit_pos = 0;
                *pos += 4;
            }

            let child = self.tree[node_idx as usize].child[bit];
            if child < -1 {
                // Leaf: encoded as -(k + 2), so symbol = -(child + 2).
                return Ok(-(child + 2));
            }
            if child < 0 {
                // -1 means no child → malformed
                return Err(LercError::DecodeFailed);
            }
            node_idx = child;
        }
    }
}
