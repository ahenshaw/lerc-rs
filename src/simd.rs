/// SIMD-accelerated dequantization for float pixel types.
///
/// Each function converts a slice of u32 quantized codes into float pixel
/// values:  `out[i] = (buf[i] as float * inv_scale + offset).min(z_max)`
///
/// The AVX2 path interprets the u32 codes as i32 via `_mm256_cvtepi32_ps` /
/// `_mm256_cvtepi32_pd`.  This is exact for quantized values < 2^31, which
/// holds for any practical LERC image (the max code is bounded by the tile's
/// data range divided by inv_scale).
///
/// Runtime feature detection is used so the binary runs correctly on CPUs
/// that don't support AVX2.
//
// ── f32 ───────────────────────────────────────────────────────────────────────
//
/// Dequantize `buf` into f32 pixels.
/// Parameters are `f64` to match the `LercScalar::dequantize_slice` signature;
/// they are narrowed to f32 before entering the hot loop.
#[inline]
pub(crate) fn dequantize_f32(
    buf: &[u32],
    out: &mut [f32],
    offset: f64,
    inv_scale: f64,
    z_max: f64,
) {
    let offset_f = offset as f32;
    let scale_f = inv_scale as f32;
    let zmax_f = z_max as f32;

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // Safety: feature detected at runtime above.
        return unsafe { dequantize_f32_avx2(buf, out, offset_f, scale_f, zmax_f) };
    }

    dequantize_f32_scalar(buf, out, offset_f, scale_f, zmax_f);
}

#[inline]
fn dequantize_f32_scalar(buf: &[u32], out: &mut [f32], offset: f32, inv_scale: f32, z_max: f32) {
    for (q, o) in buf.iter().zip(out.iter_mut()) {
        *o = (*q as f32).mul_add(inv_scale, offset).min(z_max);
    }
}

/// Process 8 f32 pixels per iteration with AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_f32_avx2(
    buf: &[u32],
    out: &mut [f32],
    offset: f32,
    inv_scale: f32,
    z_max: f32,
) {
    use std::arch::x86_64::*;

    let v_offset = _mm256_set1_ps(offset);
    let v_scale = _mm256_set1_ps(inv_scale);
    let v_zmax = _mm256_set1_ps(z_max);

    let n = buf.len();
    let mut i = 0;

    while i + 8 <= n {
        unsafe {
            // Load 8 × u32; reinterpret as i32 for cvtepi32_ps (valid when < 2^31).
            let qi = _mm256_loadu_si256(buf.as_ptr().add(i) as *const __m256i);
            let qf = _mm256_cvtepi32_ps(qi);
            // qf * scale + offset
            let z = _mm256_add_ps(_mm256_mul_ps(qf, v_scale), v_offset);
            let clamped = _mm256_min_ps(z, v_zmax);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), clamped);
        }
        i += 8;
    }

    // Scalar tail for the remainder (0–7 elements).
    dequantize_f32_scalar(&buf[i..], &mut out[i..], offset, inv_scale, z_max);
}

// ── f64 ───────────────────────────────────────────────────────────────────────

/// Dequantize `buf` into f64 pixels.
#[inline]
pub(crate) fn dequantize_f64(
    buf: &[u32],
    out: &mut [f64],
    offset: f64,
    inv_scale: f64,
    z_max: f64,
) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { dequantize_f64_avx2(buf, out, offset, inv_scale, z_max) };
    }

    dequantize_f64_scalar(buf, out, offset, inv_scale, z_max);
}

#[inline]
fn dequantize_f64_scalar(buf: &[u32], out: &mut [f64], offset: f64, inv_scale: f64, z_max: f64) {
    for (q, o) in buf.iter().zip(out.iter_mut()) {
        *o = (*q as f64).mul_add(inv_scale, offset).min(z_max);
    }
}

/// Process 4 f64 pixels per iteration with AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_f64_avx2(
    buf: &[u32],
    out: &mut [f64],
    offset: f64,
    inv_scale: f64,
    z_max: f64,
) {
    use std::arch::x86_64::*;

    let v_offset = _mm256_set1_pd(offset);
    let v_scale = _mm256_set1_pd(inv_scale);
    let v_zmax = _mm256_set1_pd(z_max);

    let n = buf.len();
    let mut i = 0;

    while i + 4 <= n {
        unsafe {
            // Load 4 × u32 into a 128-bit register; cvtepi32_pd extends to 4 × f64.
            let qi_128 = _mm_loadu_si128(buf.as_ptr().add(i) as *const __m128i);
            let qd = _mm256_cvtepi32_pd(qi_128);
            let z = _mm256_add_pd(_mm256_mul_pd(qd, v_scale), v_offset);
            let clamped = _mm256_min_pd(z, v_zmax);
            _mm256_storeu_pd(out.as_mut_ptr().add(i), clamped);
        }
        i += 4;
    }

    dequantize_f64_scalar(&buf[i..], &mut out[i..], offset, inv_scale, z_max);
}
