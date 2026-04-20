/// Decode-speed comparison: lerc-rs (pure Rust) vs lerc (C FFI via lerc-ref).
///
/// Each fixture is encoded once at startup with the reference crate, then
/// both decoders are timed against the same blob.
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lerc_ref as ref_lerc;
use lerc_howell as howell;

/// Build a LERC1 (CntZImage) blob for a 1-tile all-valid f32 image.
/// Mirrors the C++ BitStuff_Before_Lerc2v3 format: pack MSB-first into
/// uint32 words, right-shift the last word by ntbnn*8 to place data in
/// the low bytes, write as LE.
fn make_lerc1_blob_f32(width: usize, height: usize, pixels: &[f32], max_z_error: f64) -> Vec<u8> {
    let z_min = pixels.iter().copied().fold(f32::INFINITY, f32::min);
    let z_max = pixels.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let inv_scale = 2.0 * max_z_error;
    let n = width * height;

    let quant: Vec<u32> = pixels
        .iter()
        .map(|&v| ((v - z_min) / inv_scale as f32).round() as u32)
        .collect();
    let max_q = *quant.iter().max().unwrap_or(&0);
    let num_bits = if max_q == 0 { 0 } else { (u32::BITS - max_q.leading_zeros()) as usize };

    // Pack values MSB-first into uint32 words (C++ BitStuff_Before_Lerc2v3 format).
    let packed = if num_bits > 0 {
        let total_bits = n * num_bits;
        let n_words = total_bits.div_ceil(32);
        let mut words = vec![0u32; n_words];
        let mut bit_pos: i32 = 0;
        let mut wi = 0usize;
        for &q in &quant {
            let rem = 32 - bit_pos;
            if rem >= num_bits as i32 {
                words[wi] |= q << (rem - num_bits as i32);
                bit_pos += num_bits as i32;
                if bit_pos == 32 { bit_pos = 0; wi += 1; }
            } else {
                let ov = num_bits as i32 - rem;
                words[wi] |= q >> ov; wi += 1;
                words[wi] |= q << (32 - ov);
                bit_pos = ov;
            }
        }
        let tail = total_bits & 31;
        let ntbnn = if tail > 0 { 4 - (tail + 7) / 8 } else { 0 };
        if ntbnn > 0 { words[n_words - 1] >>= ntbnn * 8; }
        let n_bytes = n_words * 4 - ntbnn;
        let mut out = Vec::with_capacity(n_bytes);
        for w in &words { out.extend_from_slice(&w.to_le_bytes()); }
        out.truncate(n_bytes);
        out
    } else {
        vec![]
    };

    // Bitstuffer block: [hdr(1) | numElems(4) | packed]
    let mut bs = vec![num_bits as u8];
    bs.extend_from_slice(&(n as u32).to_le_bytes());
    bs.extend_from_slice(&packed);

    // Z-tile: raw_flag=1 (bit-stuffed, 4-byte f32 offset)
    let mut z_tile = vec![0x01u8];
    z_tile.extend_from_slice(&z_min.to_le_bytes());
    z_tile.extend_from_slice(&bs);
    let z_bytes = z_tile.len();

    let mut blob = Vec::new();
    blob.extend_from_slice(b"CntZImage ");
    blob.extend_from_slice(&11i32.to_le_bytes());
    blob.extend_from_slice(&8i32.to_le_bytes());
    blob.extend_from_slice(&(height as i32).to_le_bytes());
    blob.extend_from_slice(&(width as i32).to_le_bytes());
    blob.extend_from_slice(&max_z_error.to_le_bytes());
    // Cnt part: constant all-valid (numTilesVert=0, numTilesHori=0, numBytes=0, maxVal=1.0)
    blob.extend_from_slice(&0i32.to_le_bytes());
    blob.extend_from_slice(&0i32.to_le_bytes());
    blob.extend_from_slice(&0i32.to_le_bytes());
    blob.extend_from_slice(&1.0f32.to_le_bytes());
    // Z part: 1×1 tile
    blob.extend_from_slice(&1i32.to_le_bytes());
    blob.extend_from_slice(&1i32.to_le_bytes());
    blob.extend_from_slice(&(z_bytes as i32).to_le_bytes());
    blob.extend_from_slice(&z_max.to_le_bytes());
    blob.extend_from_slice(&z_tile);
    blob
}

fn make_blob<T: ref_lerc::LercDataType>(
    data: &[T],
    width: usize,
    height: usize,
    n_bands: usize,
    max_z_error: f64,
) -> Vec<u8> {
    ref_lerc::encode(data, None, width, height, 1, n_bands, 0, max_z_error)
        .expect("reference encode failed")
}

/// Build a LERC1 blob using raw f32 storage (compr_flag=0), the lossless path.
fn make_lerc1_blob_f32_lossless(width: usize, height: usize, pixels: &[f32]) -> Vec<u8> {
    let z_max = pixels.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let n = width * height;

    // Z-tile: raw_flag=0, one f32 per pixel.
    let mut z_tile = vec![0x00u8];
    for &v in pixels {
        z_tile.extend_from_slice(&v.to_le_bytes());
    }
    let z_bytes = z_tile.len();

    let mut blob = Vec::new();
    blob.extend_from_slice(b"CntZImage ");
    blob.extend_from_slice(&11i32.to_le_bytes());
    blob.extend_from_slice(&8i32.to_le_bytes());
    blob.extend_from_slice(&(height as i32).to_le_bytes());
    blob.extend_from_slice(&(width as i32).to_le_bytes());
    blob.extend_from_slice(&0.0f64.to_le_bytes()); // max_z_error = 0 (lossless)
    // Cnt part: constant all-valid
    blob.extend_from_slice(&0i32.to_le_bytes());
    blob.extend_from_slice(&0i32.to_le_bytes());
    blob.extend_from_slice(&0i32.to_le_bytes());
    blob.extend_from_slice(&1.0f32.to_le_bytes());
    // Z part: 1×1 tile
    blob.extend_from_slice(&1i32.to_le_bytes());
    blob.extend_from_slice(&1i32.to_le_bytes());
    blob.extend_from_slice(&(z_bytes as i32).to_le_bytes());
    blob.extend_from_slice(&z_max.to_le_bytes());
    blob.extend_from_slice(&z_tile);
    let _ = n;
    blob
}

fn bench_decode(c: &mut Criterion) {
    // -----------------------------------------------------------------
    // Fixtures
    // -----------------------------------------------------------------

    // Small u8 (256 px) – tests per-call overhead.
    let (w_sm, h_sm) = (16usize, 16usize);
    let pixels_u8_sm: Vec<u8> = (0..w_sm * h_sm).map(|i| (i * 7 % 256) as u8).collect();
    let blob_u8_sm = make_blob(&pixels_u8_sm, w_sm, h_sm, 1, 0.0);

    // Large u8 (1 MP).
    let (w_lg, h_lg) = (1024usize, 1024usize);
    let pixels_u8_lg: Vec<u8> = (0..w_lg * h_lg).map(|i| (i * 7 % 256) as u8).collect();
    let blob_u8_lg = make_blob(&pixels_u8_lg, w_lg, h_lg, 1, 0.0);

    // Large i16.
    let pixels_i16_lg: Vec<i16> = (0..w_lg * h_lg)
        .map(|i| (i as i32 * 17 - 32768) as i16)
        .collect();
    let blob_i16_lg = make_blob(&pixels_i16_lg, w_lg, h_lg, 1, 0.0);

    // Large f32 (lossy).
    let pixels_f32_lg: Vec<f32> = (0..w_lg * h_lg).map(|i| (i as f32) * 0.3).collect();
    let blob_f32_lg = make_blob(&pixels_f32_lg, w_lg, h_lg, 1, 0.5);

    // Large f32 (lossless – DeltaDeltaHuffman path).
    let pixels_f32_ll: Vec<f32> = (0..w_lg * h_lg)
        .map(|i| (i as f32 * 1.234_567_8).sin() * 1000.0)
        .collect();
    let blob_f32_ll = make_blob(&pixels_f32_ll, w_lg, h_lg, 1, 0.0);

    // Large f64 (lossy – tiled path).
    let pixels_f64_lg: Vec<f64> = (0..w_lg * h_lg).map(|i| (i as f64) * 0.3).collect();
    let blob_f64_lg = make_blob(&pixels_f64_lg, w_lg, h_lg, 1, 0.5);

    // 3-band u8 (1 MP × 3 bands).
    let pixels_u8_3band: Vec<u8> = (0..w_lg * h_lg * 3).map(|i| (i * 11 % 256) as u8).collect();
    let blob_u8_3band = make_blob(&pixels_u8_3band, w_lg, h_lg, 3, 0.0);

    // LERC1 f32 1 MP (lossy).
    let pixels_lerc1: Vec<f32> = (0..w_lg * h_lg).map(|i| (i as f32) * 0.3).collect();
    let blob_lerc1 = make_lerc1_blob_f32(w_lg, h_lg, &pixels_lerc1, 0.5);

    // LERC1 f32 1 MP (lossless – raw f32 path).
    let pixels_lerc1_ll: Vec<f32> = (0..w_lg * h_lg)
        .map(|i| (i as f32 * 1.234_567_8).sin() * 1000.0)
        .collect();
    let blob_lerc1_ll = make_lerc1_blob_f32_lossless(w_lg, h_lg, &pixels_lerc1_ll);

    // -----------------------------------------------------------------
    // u8 small
    // -----------------------------------------------------------------
    let mut g = c.benchmark_group("u8_small_16x16");
    g.bench_function("lerc-rs", |b| {
        b.iter(|| lerc::decode(black_box(&blob_u8_sm)).unwrap())
    });
    g.bench_function("lerc-ref", |b| {
        b.iter(|| ref_lerc::decode_auto::<u8>(black_box(&blob_u8_sm)).unwrap())
    });
    g.bench_function("lerc-howell", |b| {
        b.iter(|| howell::decode_slice::<u8>(black_box(&blob_u8_sm)).unwrap())
    });
    g.finish();

    // -----------------------------------------------------------------
    // u8 1 MP
    // -----------------------------------------------------------------
    let mut g = c.benchmark_group("u8_1mp_1024x1024");
    g.bench_function("lerc-rs", |b| {
        b.iter(|| lerc::decode(black_box(&blob_u8_lg)).unwrap())
    });
    g.bench_function("lerc-ref", |b| {
        b.iter(|| ref_lerc::decode_auto::<u8>(black_box(&blob_u8_lg)).unwrap())
    });
    g.bench_function("lerc-howell", |b| {
        b.iter(|| howell::decode_slice::<u8>(black_box(&blob_u8_lg)).unwrap())
    });
    g.finish();

    // -----------------------------------------------------------------
    // i16 1 MP
    // -----------------------------------------------------------------
    let mut g = c.benchmark_group("i16_1mp_1024x1024");
    g.bench_function("lerc-rs", |b| {
        b.iter(|| lerc::decode(black_box(&blob_i16_lg)).unwrap())
    });
    g.bench_function("lerc-ref", |b| {
        b.iter(|| ref_lerc::decode_auto::<i16>(black_box(&blob_i16_lg)).unwrap())
    });
    g.bench_function("lerc-howell", |b| {
        b.iter(|| howell::decode_slice::<i16>(black_box(&blob_i16_lg)).unwrap())
    });
    g.finish();

    // -----------------------------------------------------------------
    // f32 1 MP (lossy)
    // -----------------------------------------------------------------
    let mut g = c.benchmark_group("f32_1mp_lossy_1024x1024");
    g.bench_function("lerc-rs", |b| {
        b.iter(|| lerc::decode(black_box(&blob_f32_lg)).unwrap())
    });
    g.bench_function("lerc-ref", |b| {
        b.iter(|| ref_lerc::decode_auto::<f32>(black_box(&blob_f32_lg)).unwrap())
    });
    g.bench_function("lerc-howell", |b| {
        b.iter(|| howell::decode_slice::<f32>(black_box(&blob_f32_lg)).unwrap())
    });
    g.finish();

    // -----------------------------------------------------------------
    // f32 1 MP (lossless)
    // -----------------------------------------------------------------
    let mut g = c.benchmark_group("f32_1mp_lossless_1024x1024");
    g.bench_function("lerc-rs", |b| {
        b.iter(|| lerc::decode(black_box(&blob_f32_ll)).unwrap())
    });
    g.bench_function("lerc-ref", |b| {
        b.iter(|| ref_lerc::decode_auto::<f32>(black_box(&blob_f32_ll)).unwrap())
    });
    g.bench_function("lerc-howell", |b| {
        b.iter(|| howell::decode_slice::<f32>(black_box(&blob_f32_ll)).unwrap())
    });
    g.finish();

    // -----------------------------------------------------------------
    // f64 1 MP (lossy)
    // -----------------------------------------------------------------
    let mut g = c.benchmark_group("f64_1mp_lossy_1024x1024");
    g.bench_function("lerc-rs", |b| {
        b.iter(|| lerc::decode(black_box(&blob_f64_lg)).unwrap())
    });
    g.bench_function("lerc-ref", |b| {
        b.iter(|| ref_lerc::decode_auto::<f64>(black_box(&blob_f64_lg)).unwrap())
    });
    g.bench_function("lerc-howell", |b| {
        b.iter(|| howell::decode_slice::<f64>(black_box(&blob_f64_lg)).unwrap())
    });
    g.finish();

    // -----------------------------------------------------------------
    // u8 3-band 1 MP
    // -----------------------------------------------------------------
    let mut g = c.benchmark_group("u8_3band_1mp_1024x1024");
    g.bench_function("lerc-rs", |b| {
        b.iter(|| lerc::decode(black_box(&blob_u8_3band)).unwrap())
    });
    g.bench_function("lerc-ref", |b| {
        b.iter(|| ref_lerc::decode_auto::<u8>(black_box(&blob_u8_3band)).unwrap())
    });
    g.bench_function("lerc-howell", |b| {
        b.iter(|| howell::decode(black_box(&blob_u8_3band)).unwrap())
    });
    g.finish();

    // -----------------------------------------------------------------
    // LERC1 f32 1 MP (lossless)
    // -----------------------------------------------------------------
    let mut g = c.benchmark_group("lerc1_f32_1mp_lossless_1024x1024");
    g.bench_function("lerc-rs", |b| {
        b.iter(|| lerc::decode(black_box(&blob_lerc1_ll)).unwrap())
    });
    g.bench_function("lerc-ref", |b| {
        b.iter(|| ref_lerc::decode_auto::<f32>(black_box(&blob_lerc1_ll)).unwrap())
    });
    g.bench_function("lerc-howell", |b| {
        b.iter(|| howell::decode_slice::<f32>(black_box(&blob_lerc1_ll)).unwrap())
    });
    g.finish();

    // -----------------------------------------------------------------
    // LERC1 f32 1 MP (lossy)
    // -----------------------------------------------------------------
    let mut g = c.benchmark_group("lerc1_f32_1mp_lossy_1024x1024");
    g.bench_function("lerc-rs", |b| {
        b.iter(|| lerc::decode(black_box(&blob_lerc1)).unwrap())
    });
    g.bench_function("lerc-ref", |b| {
        b.iter(|| ref_lerc::decode_auto::<f32>(black_box(&blob_lerc1)).unwrap())
    });
    g.bench_function("lerc-howell", |b| {
        b.iter(|| howell::decode_slice::<f32>(black_box(&blob_lerc1)).unwrap())
    });
    g.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
