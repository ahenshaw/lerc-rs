/// Compact bit array where bit `k` = 1 means pixel `k` (linear index) is valid.
/// Matches the layout of the C++ `BitMask` class: bit 7 of byte 0 is pixel 0.
pub(crate) struct BitMask {
    data: Vec<u8>,
    width: i32,
    height: i32,
}

#[allow(dead_code)]
impl BitMask {
    pub fn new(width: i32, height: i32) -> Option<Self> {
        if width <= 0 || height <= 0 {
            return None;
        }
        let n_pixels = width as usize * height as usize;
        let size = (n_pixels + 7) / 8;
        Some(Self {
            data: vec![0u8; size],
            width,
            height,
        })
    }

    pub fn width(&self) -> i32 {
        self.width
    }

    pub fn size(&self) -> usize {
        (self.width as usize * self.height as usize + 7) / 8
    }

    pub fn n_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }

    pub fn set_all_valid(&mut self) {
        self.data.fill(0xFF);
    }

    pub fn set_all_invalid(&mut self) {
        self.data.fill(0x00);
    }

    #[inline]
    pub fn is_valid(&self, k: i32) -> bool {
        let k = k as usize;
        let byte_idx = k >> 3;
        let bit = 1u8 << (7 - (k & 7));
        byte_idx < self.data.len() && (self.data[byte_idx] & bit) != 0
    }

    pub fn count_valid_bits(&self) -> i32 {
        let n_pix = self.width as usize * self.height as usize;
        let full_bytes = n_pix / 8;
        let rem_bits = n_pix % 8;
        let mut cnt: i32 = self.data[..full_bytes]
            .iter()
            .map(|b| b.count_ones() as i32)
            .sum();
        if rem_bits > 0 {
            // Only count the top `rem_bits` bits (bit 7 = pixel 0 within byte).
            let last = self.data[full_bytes];
            for bit in 0..rem_bits {
                cnt += ((last >> (7 - bit)) & 1) as i32;
            }
        }
        cnt
    }

    pub fn bits(&self) -> &[u8] {
        &self.data
    }

    pub fn bits_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}
