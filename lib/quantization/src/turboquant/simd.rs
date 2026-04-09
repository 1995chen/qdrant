//! Small SIMD helpers for the scoring path.
//!
//! The TurboQuant encoding itself stays identical. SIMD is only used to speed
//! up the final dot product against the reconstructed vector. This keeps the
//! implementation easy to audit while still giving us an explicit AVX2 /
//! AVX512 / NEON variant for experiments.

#[inline]
pub fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { dot_avx512(lhs, rhs) };
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { dot_avx2(lhs, rhs) };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            return unsafe { dot_neon(lhs, rhs) };
        }
    }

    dot_plain(lhs, rhs)
}

fn dot_plain(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(&a, &b)| a * b).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(lhs: &[f32], rhs: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut index = 0usize;
    while index + 8 <= lhs.len() {
        let a = unsafe { _mm256_loadu_ps(lhs.as_ptr().add(index)) };
        let b = unsafe { _mm256_loadu_ps(rhs.as_ptr().add(index)) };
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
        index += 8;
    }

    let mut lanes = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
    let mut total: f32 = lanes.into_iter().sum();
    total += dot_plain(&lhs[index..], &rhs[index..]);
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(lhs: &[f32], rhs: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm512_setzero_ps();
    let mut index = 0usize;
    while index + 16 <= lhs.len() {
        let a = unsafe { _mm512_loadu_ps(lhs.as_ptr().add(index)) };
        let b = unsafe { _mm512_loadu_ps(rhs.as_ptr().add(index)) };
        sum = _mm512_add_ps(sum, _mm512_mul_ps(a, b));
        index += 16;
    }

    let mut lanes = [0.0f32; 16];
    unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), sum) };
    let mut total: f32 = lanes.into_iter().sum();
    total += dot_plain(&lhs[index..], &rhs[index..]);
    total
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn dot_neon(lhs: &[f32], rhs: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let mut sum = vdupq_n_f32(0.0);
    let mut index = 0usize;
    while index + 4 <= lhs.len() {
        let a = unsafe { vld1q_f32(lhs.as_ptr().add(index)) };
        let b = unsafe { vld1q_f32(rhs.as_ptr().add(index)) };
        sum = vaddq_f32(sum, vmulq_f32(a, b));
        index += 4;
    }

    let mut total = vaddvq_f32(sum);
    total += dot_plain(&lhs[index..], &rhs[index..]);
    total
}
