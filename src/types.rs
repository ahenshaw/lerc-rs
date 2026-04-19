/// Data types supported by LERC, matching the C++ enum order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DataType {
    I8 = 0,
    U8 = 1,
    I16 = 2,
    U16 = 3,
    I32 = 4,
    U32 = 5,
    F32 = 6,
    F64 = 7,
}

impl DataType {
    pub(crate) fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::I8),
            1 => Some(Self::U8),
            2 => Some(Self::I16),
            3 => Some(Self::U16),
            4 => Some(Self::I32),
            5 => Some(Self::U32),
            6 => Some(Self::F32),
            7 => Some(Self::F64),
            _ => None,
        }
    }

    /// Size of the native type in bytes.
    pub fn byte_size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// Metadata parsed from the LERC blob header(s) without fully decoding.
#[derive(Debug, Clone)]
pub struct LercInfo {
    /// LERC version: 0 = old Lerc1, 1-6 = Lerc2 v1-v6.
    pub version: i32,
    /// Number of values per pixel (depth / nDim).
    pub n_depth: i32,
    /// Number of columns.
    pub n_cols: i32,
    /// Number of rows.
    pub n_rows: i32,
    /// Number of valid pixels (for the first band).
    pub num_valid_pixel: i32,
    /// Number of bands (concatenated single-band blobs).
    pub n_bands: i32,
    /// Total blob size in bytes.
    pub blob_size: i32,
    /// Number of masks: 0 (all valid), 1 (shared), or n_bands (per-band).
    pub n_masks: i32,
    /// 0 = no noData value used; n_bands = noData used in at least one band.
    pub n_uses_no_data_value: i32,
    /// Native pixel data type.
    pub data_type: DataType,
    /// Global minimum pixel value across all bands.
    pub z_min: f64,
    /// Global maximum pixel value across all bands.
    pub z_max: f64,
    /// Maximum Z error used when encoding.
    pub max_z_error: f64,
}

/// Decoded pixel data in the native type.
///
/// Layout: `[band * n_rows * n_cols * n_depth + row * n_cols * n_depth + col * n_depth + depth]`
#[derive(Debug, Clone)]
pub enum LercData {
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// Result of [`crate::decode`].
#[derive(Debug, Clone)]
pub struct DecodedData {
    /// Decoded pixel values (see [`LercData`] for layout).
    pub data: LercData,
    /// Validity mask: `1` = valid, `0` = invalid.
    /// `None` if all pixels are valid across every band.
    /// When present, length = `n_masks * n_rows * n_cols` where masks are band-ordered.
    pub valid_pixels: Option<Vec<u8>>,
    /// Per-band noData values, length = `n_bands`.  `None` if unused.
    pub no_data_values: Option<Vec<f64>>,
    /// Header metadata for this blob.
    pub info: LercInfo,
}
