# lerc-rs

A decode-only Rust implementation of [LERC](https://github.com/Esri/lerc) (Limited Error Raster Compression), Esri's format for compressing raster and point-cloud data with a configurable per-pixel error bound.

No external dependencies.

## Supported formats

- LERC2 versions 1–6
- All eight pixel types: `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `f32`, `f64`
- Single-band and multi-band images
- Validity masks
- Lossy and lossless (integer) encoding

**Not supported:** Lerc1 (CntZImage), lossless float (v6 DeltaDeltaHuffman).

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
| `version` | `i32` | LERC2 version (1–6) |
| `n_cols` | `i32` | Image width |
| `n_rows` | `i32` | Image height |
| `n_depth` | `i32` | Values per pixel (spectral depth) |
| `n_bands` | `i32` | Number of bands |
| `data_type` | `DataType` | Native pixel type |
| `num_valid_pixel` | `i32` | Valid pixel count for the first band |
| `z_min` / `z_max` | `f64` | Global pixel value range |
| `max_z_error` | `f64` | Per-pixel error bound used during encoding |
| `blob_size` | `i32` | Total blob size in bytes |
