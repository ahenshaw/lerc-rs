pub mod error;
pub mod types;

mod bitmask;
mod bitstuffer;
mod huffman;
mod lerc2;
mod lossless_float;
mod rle;
mod simd;

pub use error::LercError;
pub use types::{DataType, DecodedData, LercData, LercInfo};

/// Decode a LERC blob into pixel data, validity mask, and metadata.
///
/// # Errors
/// Returns `LercError` on malformed, truncated, or unsupported blobs.
pub fn decode(src: &[u8]) -> Result<DecodedData, LercError> {
    lerc2::decode(src)
}

/// Parse metadata from a LERC blob without decoding pixel data.
///
/// # Errors
/// Returns `LercError` on malformed or unsupported blobs.
pub fn get_lerc_info(src: &[u8]) -> Result<LercInfo, LercError> {
    lerc2::get_lerc_info(src)
}
