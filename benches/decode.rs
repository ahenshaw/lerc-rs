/// Decode-speed comparison: lerc-rs (pure Rust) vs lerc (C FFI via lerc-ref).
///
/// Each fixture is encoded once at startup with the reference crate, then
/// both decoders are timed against the same blob.
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lerc_ref as ref_lerc;

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
    let pixels_i16_lg: Vec<i16> =
        (0..w_lg * h_lg).map(|i| (i as i32 * 17 - 32768) as i16).collect();
    let blob_i16_lg = make_blob(&pixels_i16_lg, w_lg, h_lg, 1, 0.0);

    // Large f32 (lossy – avoids unsupported DeltaDeltaHuffman path).
    let pixels_f32_lg: Vec<f32> = (0..w_lg * h_lg).map(|i| (i as f32) * 0.3).collect();
    let blob_f32_lg = make_blob(&pixels_f32_lg, w_lg, h_lg, 1, 0.5);

    // Large f64 (lossy – tiled path).
    let pixels_f64_lg: Vec<f64> = (0..w_lg * h_lg).map(|i| (i as f64) * 0.3).collect();
    let blob_f64_lg = make_blob(&pixels_f64_lg, w_lg, h_lg, 1, 0.5);

    // 3-band u8 (1 MP × 3 bands).
    let pixels_u8_3band: Vec<u8> =
        (0..w_lg * h_lg * 3).map(|i| (i * 11 % 256) as u8).collect();
    let blob_u8_3band = make_blob(&pixels_u8_3band, w_lg, h_lg, 3, 0.0);

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
    g.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
