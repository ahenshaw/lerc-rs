/// Integration tests that encode with the reference `lerc` crate and decode
/// with this library, comparing pixel data, validity masks, and metadata.
use lerc_ref as ref_lerc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Encode data with the reference crate; panic on error.
fn ref_encode<T: ref_lerc::LercDataType>(
    data: &[T],
    mask: Option<&[u8]>,
    width: usize,
    height: usize,
    depth: usize,
    n_bands: usize,
    max_z_error: f64,
) -> Vec<u8> {
    let n_masks = if mask.is_some() { 1 } else { 0 };
    ref_lerc::encode(
        data,
        mask,
        width,
        height,
        depth,
        n_bands,
        n_masks,
        max_z_error,
    )
    .expect("reference encode failed")
}

/// Assert two float slices agree to within `tol`.
fn assert_approx_eq_f32(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "slice length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "f32 mismatch at index {i}: {x} vs {y} (tol={tol})"
        );
    }
}

fn assert_approx_eq_f64(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "slice length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "f64 mismatch at index {i}: {x} vs {y} (tol={tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Dump blob header bytes for debugging.
#[test]
#[allow(clippy::cast_sign_loss)]
fn debug_blob_header() {
    let width = 8usize;
    let height = 6usize;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i * 7 % 256) as u8).collect();
    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);
    println!("blob len = {}", blob.len());
    println!("first 100 bytes: {:02x?}", &blob[..100.min(blob.len())]);
    let ver = i32::from_le_bytes(blob[6..10].try_into().unwrap());
    println!("version = {ver}");
    let checksum = u32::from_le_bytes(blob[10..14].try_into().unwrap());
    println!("checksum = {checksum:#010x}");
    let info = ref_lerc::get_blob_info(&blob).unwrap();
    println!("ref info: {:?}", info);
    // Print full blob hex with per-byte annotations
    println!("full blob ({} bytes):", blob.len());
    for (i, b) in blob.iter().enumerate() {
        print!("[{i:3}]={b:02x}  ");
        if (i + 1) % 8 == 0 {
            println!();
        }
    }
    println!();
}

/// Lossless u8 image, all pixels valid.
#[test]
fn u8_all_valid_lossless() {
    let width = 8usize;
    let height = 6usize;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i * 7 % 256) as u8).collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::U8(out) = result.data else {
        panic!("expected U8 data, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
    assert!(
        result.valid_pixels.is_none(),
        "expected no mask for all-valid"
    );
}

/// Lossless u8 image with some invalid pixels (validity mask).
#[test]
fn u8_with_mask_lossless() {
    let width = 10usize;
    let height = 8usize;
    let n = width * height;
    let pixels: Vec<u8> = (0..n).map(|i| (i % 200) as u8).collect();
    // Mark every third pixel invalid.
    let mask: Vec<u8> = (0..n).map(|i| if i % 3 == 0 { 0 } else { 1 }).collect();

    let blob = ref_encode(&pixels, Some(&mask), width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::U8(out) = result.data else {
        panic!("expected U8, got {:?}", result.info.data_type);
    };
    // Valid pixels must match exactly.
    for i in 0..n {
        if mask[i] == 1 {
            assert_eq!(out[i], pixels[i], "mismatch at valid pixel {i}");
        }
    }
    // The returned mask must agree with the input mask.
    let vm = result.valid_pixels.expect("expected a mask in output");
    assert_eq!(vm.len(), n, "mask length mismatch");
    for i in 0..n {
        assert_eq!(vm[i], mask[i], "mask mismatch at pixel {i}");
    }
}

/// Lossless i8 image.
#[test]
fn i8_all_valid_lossless() {
    let width = 7usize;
    let height = 5usize;
    let pixels: Vec<i8> = (0..width * height)
        .map(|i| (i as i32 * 3 - 64) as i8)
        .collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::I8(out) = result.data else {
        panic!("expected I8, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
}

/// Lossless i16 image.
#[test]
fn i16_all_valid_lossless() {
    let width = 12usize;
    let height = 10usize;
    let pixels: Vec<i16> = (0..width * height)
        .map(|i| (i as i32 * 17 - 500) as i16)
        .collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::I16(out) = result.data else {
        panic!("expected I16, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
}

/// Lossless u16 image.
#[test]
fn u16_all_valid_lossless() {
    let width = 9usize;
    let height = 9usize;
    let pixels: Vec<u16> = (0..width * height).map(|i| (i * 31) as u16).collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::U16(out) = result.data else {
        panic!("expected U16, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
}

/// Lossless i32 image.
#[test]
fn i32_all_valid_lossless() {
    let width = 5usize;
    let height = 5usize;
    let pixels: Vec<i32> = (0..width * height)
        .map(|i| (i as i32 * 1000) - 12000)
        .collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::I32(out) = result.data else {
        panic!("expected I32, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
}

/// Lossless f32 image (DeltaDeltaHuffman path, max_z_error=0.0).
#[test]
fn f32_all_valid_lossless() {
    let width = 8usize;
    let height = 8usize;
    let pixels: Vec<f32> = (0..width * height)
        .map(|i| (i as f32) * 0.5 - 16.0)
        .collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::F32(out) = result.data else {
        panic!("expected F32, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels, "lossless f32 round-trip failed");
}

/// Lossless f32 with non-trivial values (tests DeltaDeltaHuffman predictor paths).
#[test]
fn f32_lossless_varied() {
    let width = 32usize;
    let height = 32usize;
    let pixels: Vec<f32> = (0..width * height)
        .map(|i| {
            let x = i as f32;
            (x * 1.234_567_8).sin() * 100.0 + x * 0.01
        })
        .collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::F32(out) = result.data else {
        panic!("expected F32, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels, "lossless f32 round-trip failed");
}

/// Lossy f32 image with max_z_error = 0.5.
#[test]
fn f32_lossy() {
    let width = 16usize;
    let height = 16usize;
    let max_z_error = 0.5f64;
    let pixels: Vec<f32> = (0..width * height).map(|i| (i as f32) * 0.3).collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, max_z_error);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::F32(out) = result.data else {
        panic!("expected F32, got {:?}", result.info.data_type);
    };
    assert_approx_eq_f32(&out, &pixels, max_z_error as f32);
}

/// Lossless f64 image.
#[test]
fn f64_all_valid_lossless() {
    let width = 6usize;
    let height = 6usize;
    let pixels: Vec<f64> = (0..width * height)
        .map(|i| (i as f64) * 1.23456789)
        .collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::F64(out) = result.data else {
        panic!("expected F64, got {:?}", result.info.data_type);
    };
    assert_approx_eq_f64(&out, &pixels, 0.0);
}

/// Multi-band u8 image (3 bands).
#[test]
fn u8_multiband() {
    let width = 8usize;
    let height = 8usize;
    let n_bands = 3usize;
    let n = width * height * n_bands;
    let pixels: Vec<u8> = (0..n).map(|i| (i * 11 % 256) as u8).collect();

    let blob = ref_encode(&pixels, None, width, height, 1, n_bands, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::U8(out) = result.data else {
        panic!("expected U8, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
    assert_eq!(result.info.n_bands, n_bands as i32);
}

/// Metadata from get_lerc_info matches what was encoded.
#[test]
fn get_lerc_info_u8() {
    let width = 10usize;
    let height = 7usize;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 200) as u8).collect();
    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);

    let info = lerc::get_lerc_info(&blob).expect("get_lerc_info failed");
    assert_eq!(info.n_cols, width as i32);
    assert_eq!(info.n_rows, height as i32);
    assert_eq!(info.n_bands, 1);
    assert_eq!(info.data_type, lerc::DataType::U8);
    assert_eq!(info.num_valid_pixel, (width * height) as i32);
}

/// get_lerc_info with a mask reports the correct valid pixel count.
#[test]
fn get_lerc_info_with_mask() {
    let width = 8usize;
    let height = 8usize;
    let n = width * height;
    let pixels: Vec<u8> = (0..n).map(|i| i as u8).collect();
    // Half the pixels are valid.
    let mask: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
    let n_valid = mask.iter().filter(|&&b| b == 1).count();

    let blob = ref_encode(&pixels, Some(&mask), width, height, 1, 1, 0.0);
    let info = lerc::get_lerc_info(&blob).expect("get_lerc_info failed");

    assert_eq!(info.n_cols, width as i32);
    assert_eq!(info.n_rows, height as i32);
    assert_eq!(info.num_valid_pixel, n_valid as i32);
}

/// A uniform image (all pixels the same value) compresses to a const tile
/// and should decode cleanly.
#[test]
fn u8_uniform_image() {
    let width = 16usize;
    let height = 16usize;
    let pixels = vec![42u8; width * height];

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::U8(out) = result.data else {
        panic!("expected U8, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
}

/// 1×1 image edge case.
#[test]
fn single_pixel() {
    let pixels = vec![200u8; 1];
    let blob = ref_encode(&pixels, None, 1, 1, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::U8(out) = result.data else {
        panic!("expected U8, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
}

// ---------------------------------------------------------------------------
// Data-type coverage
// ---------------------------------------------------------------------------

/// Lossless u32 image — the only data type with no other test coverage.
#[test]
fn u32_all_valid_lossless() {
    let width = 8usize;
    let height = 8usize;
    let pixels: Vec<u32> = (0..width * height).map(|i| i as u32 * 12345).collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::U32(out) = result.data else {
        panic!("expected U32, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
}

/// Lossless f64 with non-trivial values (bit-exact round-trip check).
/// Complements `f64_all_valid_lossless` which only uses simple linear values.
#[test]
fn f64_lossless_varied() {
    let width = 32usize;
    let height = 32usize;
    let pixels: Vec<f64> = (0..width * height)
        .map(|i| (i as f64 * 1.234_567_890_123_4).sin() * 1000.0)
        .collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::F64(out) = result.data else {
        panic!("expected F64, got {:?}", result.info.data_type);
    };
    for (i, (&a, &b)) in out.iter().zip(pixels.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "bit mismatch at index {i}: {a} vs {b}"
        );
    }
}

/// Lossy f64 — only lossless was tested before.
#[test]
fn f64_lossy() {
    let width = 16usize;
    let height = 16usize;
    let max_z_error = 0.5f64;
    let pixels: Vec<f64> = (0..width * height).map(|i| i as f64 * 0.3).collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, max_z_error);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::F64(out) = result.data else {
        panic!("expected F64, got {:?}", result.info.data_type);
    };
    // Add a small epsilon: floating-point subtraction of values like 4.4 - 3.9
    // can exceed 0.5 by a ULP due to IEEE 754 rounding.
    assert_approx_eq_f64(&out, &pixels, max_z_error + 1e-10);
}

// ---------------------------------------------------------------------------
// Huffman integer path
// ---------------------------------------------------------------------------

/// u8 with max_z_error=0.5 forces the Huffman-integer encode path
/// (`try_huffman_int`).  The lossless u8 tests use max_z_error=0.0 and go
/// through the bitstuffer instead, so this path would otherwise be uncovered.
/// For u8, error=0.5 is effectively lossless (integer values round-trip exactly).
#[test]
fn u8_huffman_int_path() {
    let width = 32usize;
    let height = 32usize;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i * 7 % 256) as u8).collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.5);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::U8(out) = result.data else {
        panic!("expected U8, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
}

/// i8 with max_z_error=0.5 — same Huffman integer path, signed type.
#[test]
fn i8_huffman_int_path() {
    let width = 16usize;
    let height = 16usize;
    let pixels: Vec<i8> = (0..width * height)
        .map(|i| (i as i32 * 3 - 64) as i8)
        .collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, 0.5);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::I8(out) = result.data else {
        panic!("expected I8, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
}

// ---------------------------------------------------------------------------
// Lossy integer types
// ---------------------------------------------------------------------------

/// Lossy i16 — only f32 lossy was tested before.
#[test]
fn i16_lossy() {
    let width = 16usize;
    let height = 16usize;
    let max_z_error = 1.0f64;
    let pixels: Vec<i16> = (0..width * height)
        .map(|i| (i as i32 * 17 - 1000) as i16)
        .collect();

    let blob = ref_encode(&pixels, None, width, height, 1, 1, max_z_error);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::I16(out) = result.data else {
        panic!("expected I16, got {:?}", result.info.data_type);
    };
    for (i, (&a, &b)) in out.iter().zip(pixels.iter()).enumerate() {
        assert!(
            (a as i32 - b as i32).abs() <= max_z_error as i32,
            "i16 mismatch at {i}: {a} vs {b} (max_z_error={max_z_error})"
        );
    }
}

// ---------------------------------------------------------------------------
// Mask combinations
// ---------------------------------------------------------------------------

/// f32 lossy with a validity mask — lossy and mask had not been combined.
#[test]
fn f32_lossy_with_mask() {
    let width = 12usize;
    let height = 10usize;
    let n = width * height;
    let max_z_error = 0.5f64;
    let pixels: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let mask: Vec<u8> = (0..n).map(|i| if i % 4 == 0 { 0 } else { 1 }).collect();

    let blob = ref_encode(&pixels, Some(&mask), width, height, 1, 1, max_z_error);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::F32(out) = result.data else {
        panic!("expected F32, got {:?}", result.info.data_type);
    };
    let vm = result.valid_pixels.expect("expected mask in output");
    assert_eq!(vm.len(), n, "mask length mismatch");
    for i in 0..n {
        assert_eq!(vm[i], mask[i], "mask mismatch at {i}");
        if mask[i] == 1 {
            assert!(
                (out[i] - pixels[i]).abs() <= max_z_error as f32,
                "value mismatch at valid pixel {i}: {} vs {}",
                out[i],
                pixels[i]
            );
        }
    }
}

/// i16 lossless with a mask — mask was previously only tested for u8.
#[test]
fn i16_with_mask() {
    let width = 10usize;
    let height = 8usize;
    let n = width * height;
    let pixels: Vec<i16> = (0..n).map(|i| (i as i16) * 100 - 4000).collect();
    let mask: Vec<u8> = (0..n).map(|i| if i % 5 == 0 { 0 } else { 1 }).collect();

    let blob = ref_encode(&pixels, Some(&mask), width, height, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::I16(out) = result.data else {
        panic!("expected I16, got {:?}", result.info.data_type);
    };
    let vm = result.valid_pixels.expect("expected mask in output");
    assert_eq!(vm.len(), n, "mask length mismatch");
    for i in 0..n {
        assert_eq!(vm[i], mask[i], "mask mismatch at {i}");
        if mask[i] == 1 {
            assert_eq!(out[i], pixels[i], "value mismatch at valid pixel {i}");
        }
    }
}

/// Multi-band image with a shared validity mask.
#[test]
fn multiband_with_mask() {
    let width = 8usize;
    let height = 8usize;
    let n_bands = 2usize;
    let n = width * height;
    let pixels: Vec<u8> = (0..n * n_bands).map(|i| (i * 13 % 256) as u8).collect();
    let mask: Vec<u8> = (0..n).map(|i| if i % 3 == 0 { 0 } else { 1 }).collect();

    let blob = ref_encode(&pixels, Some(&mask), width, height, 1, n_bands, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::U8(out) = result.data else {
        panic!("expected U8, got {:?}", result.info.data_type);
    };
    assert_eq!(result.info.n_bands, n_bands as i32);
    let vm = result.valid_pixels.expect("expected mask in output");
    // Mask covers n_rows*n_cols per the n_masks=1 (shared) case.
    assert_eq!(vm.len(), n, "mask length mismatch");
    for i in 0..n {
        assert_eq!(vm[i], mask[i], "mask mismatch at pixel {i}");
        if mask[i] == 1 {
            // Band-0 pixel.
            assert_eq!(out[i], pixels[i], "band-0 value mismatch at {i}");
            // Band-1 pixel (offset by n).
            assert_eq!(out[n + i], pixels[n + i], "band-1 value mismatch at {i}");
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-band float
// ---------------------------------------------------------------------------

/// f32 lossless multiband — float multiband had not been tested.
#[test]
fn f32_multiband_lossless() {
    let width = 8usize;
    let height = 8usize;
    let n_bands = 2usize;
    let n = width * height * n_bands;
    let pixels: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 32.0).collect();

    let blob = ref_encode(&pixels, None, width, height, 1, n_bands, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::F32(out) = result.data else {
        panic!("expected F32, got {:?}", result.info.data_type);
    };
    assert_eq!(result.info.n_bands, n_bands as i32);
    assert_eq!(out, pixels, "f32 multiband lossless round-trip failed");
}

// ---------------------------------------------------------------------------
// Depth (n_depth > 1)
// ---------------------------------------------------------------------------

/// Image with depth=2 (two values per pixel).  Exercises the n_depth layout
/// path: index = row * n_cols * n_depth + col * n_depth + d.
#[test]
fn depth_2_u8() {
    let width = 6usize;
    let height = 5usize;
    let depth = 2usize;
    let n = width * height * depth;
    let pixels: Vec<u8> = (0..n).map(|i| (i * 17 % 256) as u8).collect();

    let blob = ref_encode(&pixels, None, width, height, depth, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::U8(out) = result.data else {
        panic!("expected U8, got {:?}", result.info.data_type);
    };
    assert_eq!(result.info.n_depth, depth as i32);
    assert_eq!(out, pixels);
}

/// f32 lossless with depth=2 — exercises the dimension-flip in
/// `decode_lossless_f32` (iw=n_depth, ih=n_cols*n_rows when n_depth>1).
#[test]
fn depth_2_f32_lossless() {
    let width = 6usize;
    let height = 5usize;
    let depth = 2usize;
    let n = width * height * depth;
    let pixels: Vec<f32> = (0..n).map(|i| i as f32 * 1.5 - 20.0).collect();

    let blob = ref_encode(&pixels, None, width, height, depth, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");

    let lerc::LercData::F32(out) = result.data else {
        panic!("expected F32, got {:?}", result.info.data_type);
    };
    assert_eq!(result.info.n_depth, depth as i32);
    assert_eq!(out, pixels, "f32 depth=2 lossless round-trip failed");
}

// ---------------------------------------------------------------------------
// Dimension edge cases
// ---------------------------------------------------------------------------

/// 1-row image.
#[test]
fn single_row() {
    let pixels: Vec<u8> = (0..16).map(|i| (i * 11 % 256) as u8).collect();
    let blob = ref_encode(&pixels, None, 16, 1, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");
    let lerc::LercData::U8(out) = result.data else {
        panic!("expected U8, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
}

/// 1-column image.
#[test]
fn single_col() {
    let pixels: Vec<u8> = (0..16).map(|i| (i * 13 % 256) as u8).collect();
    let blob = ref_encode(&pixels, None, 1, 16, 1, 1, 0.0);
    let result = lerc::decode(&blob).expect("our decode failed");
    let lerc::LercData::U8(out) = result.data else {
        panic!("expected U8, got {:?}", result.info.data_type);
    };
    assert_eq!(out, pixels);
}

// ---------------------------------------------------------------------------
// get_lerc_info metadata fields
// ---------------------------------------------------------------------------

/// z_min, z_max, max_z_error, version, and blob_size are parsed correctly.
#[test]
fn get_lerc_info_float_metadata() {
    let width = 8usize;
    let height = 8usize;
    let max_z_error = 0.25f64;
    let pixels: Vec<f32> = (0..width * height).map(|i| i as f32 * 1.5).collect();
    let expected_min = *pixels.first().unwrap() as f64; // 0.0
    let expected_max = *pixels.last().unwrap() as f64; // 94.5

    let blob = ref_encode(&pixels, None, width, height, 1, 1, max_z_error);
    let info = lerc::get_lerc_info(&blob).expect("get_lerc_info failed");

    assert_eq!(info.data_type, lerc::DataType::F32);
    // z_min / z_max bracket the input range within the allowed error.
    assert!(
        info.z_min <= expected_min + max_z_error + 1e-9,
        "z_min {} too large (expected ≤ {})",
        info.z_min,
        expected_min + max_z_error
    );
    assert!(
        info.z_max >= expected_max - max_z_error - 1e-9,
        "z_max {} too small (expected ≥ {})",
        info.z_max,
        expected_max - max_z_error
    );
    assert!(
        info.max_z_error <= max_z_error + 1e-9,
        "max_z_error {} > {}",
        info.max_z_error,
        max_z_error
    );
    assert!(
        info.version >= 1,
        "expected LERC2 version ≥ 1, got {}",
        info.version
    );
    assert_eq!(
        info.blob_size as usize,
        blob.len(),
        "blob_size mismatch: {} vs actual {}",
        info.blob_size,
        blob.len()
    );
}
