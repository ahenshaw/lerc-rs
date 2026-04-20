# lerc-rs

A decode-only Rust implementation of [LERC](https://github.com/Esri/lerc) (Limited Error Raster Compression), Esri's format for compressing raster and point-cloud data with a configurable per-pixel error bound.

The vast majority of the code was created by an LLM converting the original C++ code.  The implementation is tested against a suite of test cases referenced against the original library.

No external dependencies.

## Supported formats

- LERC2 versions 1–6
- LERC1 (CntZImage) — decoded as `f32`
- All eight pixel types for LERC2: `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `f32`, `f64`
- Single-band and multi-band images
- Validity masks
- Lossy and lossless encoding (all types, including lossless float via DeltaDeltaHuffman)

## Usage

### Decode pixel data

```rust
let blob: Vec<u8> = std::fs::read("image.lerc2")?;

let result = lerc::decode(&blob)?;

println!(
    "{}×{}, {} band(s), type {:?}",
    result.info.n_cols, result.info.n_rows,
    result.info.n_bands, result.info.data_type,
);

match result.data {
    lerc::LercData::U8(pixels)  => { /* ... */ }
    lerc::LercData::F32(pixels) => { /* ... */ }
    // other variants for i8, i16, u16, i32, u32, f64
    _ => {}
}

// Optional validity mask: 1 = valid pixel, 0 = invalid (noData).
// None means every pixel is valid.
if let Some(mask) = result.valid_pixels {
    // mask.len() == n_rows * n_cols  (single mask) or
    //               n_bands * n_rows * n_cols  (per-band mask)
}
```

### Inspect metadata without decoding

```rust
let info = lerc::get_lerc_info(&blob)?;

println!("version:      {}", info.version);
println!("size:         {}×{}", info.n_cols, info.n_rows);
println!("bands:        {}", info.n_bands);
println!("type:         {:?}", info.data_type);
println!("valid pixels: {}", info.num_valid_pixel);
println!("z range:      [{}, {}]", info.z_min, info.z_max);
println!("max z error:  {}", info.max_z_error);
```

### Pixel layout

All pixel values are returned in a flat `Vec` with the following index order:

```
index = band * (n_rows * n_cols * n_depth)
      + row  * (n_cols * n_depth)
      + col  * n_depth
      + depth
```

For single-band, single-depth images this simplifies to `row * n_cols + col`.

### Error handling

```rust
match lerc::decode(&blob) {
    Ok(result) => { /* use result */ }
    Err(lerc::LercError::ChecksumMismatch)       => eprintln!("data corruption"),
    Err(lerc::LercError::UnsupportedVersion(v))  => eprintln!("LERC version {v} not supported"),
    Err(lerc::LercError::UnsupportedFeature(f))  => eprintln!("feature not supported: {f}"),
    Err(e)                                        => eprintln!("decode error: {e}"),
}
```

## Testing

Integration tests in `tests/reference.rs` encode with the reference C++ library (`lerc` crate v0.2.1) and decode with this library, comparing pixel values, validity masks, and metadata.

```bash
LIBCLANG_PATH=/usr/lib/llvm-18/lib cargo test --test reference
```

The `LIBCLANG_PATH` env var is required because the reference crate uses bindgen.

Coverage includes all eight pixel types (including lossy and lossless variants), validity masks, multi-band images, `n_depth > 1`, single-row/column images, all `get_lerc_info` metadata fields, and LERC1 blobs (all-valid, masked, lossy, lossless).

## Performance

Benchmarked against the reference C library (`lerc` crate v0.2.1 wrapping Esri's C++ LERC) and `lerc-rs` v0.1.1 (NathanHowell, pure Rust) on an x86-64 Linux host with AVX2, release build (`cargo bench`).

| Fixture | lerc-rs | lerc (C++) | lerc-rs v0.1.1 |
|---------|--------:|----------:|---------------:|
| u8 16×16 (per-call overhead) | 6.7 µs | 3.0 µs | 5.4 µs |
| u8 1024×1024 (1 MP) | 7.1 ms | 4.6 ms | 16.2 ms |
| i16 1024×1024 (1 MP) | 8.7 ms | 5.2 ms | 4.4 ms |
| f32 1024×1024 lossy (1 MP) | 6.7 ms | 3.6 ms | 6.3 ms |
| f32 1024×1024 lossless (1 MP) | 17.5 ms | 18.3 ms | 24.3 ms |
| f64 1024×1024 lossy (1 MP) | 14.2 ms | 4.2 ms | 6.3 ms |
| u8 1024×1024 × 3 bands | 19.2 ms | 15.1 ms | 50.4 ms |
| LERC1 f32 1024×1024 lossy | 9.3 ms | 20.1 ms | 10.4 ms |
| LERC1 f32 1024×1024 lossless | 3.3 ms | 6.6 ms | 6.0 ms |

This implementation leads on f32 lossless (LERC2 and LERC1) and is competitive on most other paths. The C++ reference holds an edge on lossy integer/float types (AVX2 SIMD) and f64 lossy decode.

## API

```rust
pub fn decode(src: &[u8]) -> Result<DecodedData, LercError>;
pub fn get_lerc_info(src: &[u8]) -> Result<LercInfo, LercError>;
```

### `DecodedData`

| Field | Type | Description |
|-------|------|-------------|
| `data` | `LercData` | Pixel values in the native type |
| `valid_pixels` | `Option<Vec<u8>>` | Validity mask (`1`=valid, `0`=invalid); `None` if all pixels are valid |
| `no_data_values` | `Option<Vec<f64>>` | Per-band noData sentinels; `None` if unused |
| `info` | `LercInfo` | Header metadata |

### `LercInfo`

| Field | Type | Description |
|-------|------|-------------|
| `version` | `i32` | LERC version (0 for LERC1, 1–6 for LERC2) |
| `n_cols` | `i32` | Image width |
| `n_rows` | `i32` | Image height |
| `n_depth` | `i32` | Values per pixel (spectral depth) |
| `n_bands` | `i32` | Number of bands |
| `data_type` | `DataType` | Native pixel type |
| `num_valid_pixel` | `i32` | Valid pixel count for the first band |
| `z_min` / `z_max` | `f64` | Global pixel value range |
| `max_z_error` | `f64` | Per-pixel error bound used during encoding |
| `blob_size` | `i32` | Total blob size in bytes |
