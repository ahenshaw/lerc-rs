use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum LercError {
    /// The data is not a valid LERC blob.
    InvalidBlob,
    /// The blob is truncated or a buffer is too small.
    TruncatedBlob,
    /// The LERC version is newer than this decoder supports.
    UnsupportedVersion(i32),
    /// A codec feature used in this blob is not yet implemented.
    UnsupportedFeature(&'static str),
    /// Fletcher-32 checksum verification failed (data corruption).
    ChecksumMismatch,
    /// Internal decode failure (malformed data).
    DecodeFailed,
}

impl fmt::Display for LercError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBlob => write!(f, "not a valid LERC blob"),
            Self::TruncatedBlob => write!(f, "truncated LERC blob"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported LERC version {v}"),
            Self::UnsupportedFeature(s) => write!(f, "unsupported feature: {s}"),
            Self::ChecksumMismatch => write!(f, "checksum mismatch"),
            Self::DecodeFailed => write!(f, "decode failed"),
        }
    }
}

impl std::error::Error for LercError {}
