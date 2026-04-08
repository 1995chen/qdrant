//! Small numeric helpers used by the Lloyd-Max codebook builder.
//!
//! We avoid pulling in a heavyweight statistics crate because the standalone
//! TurboQuant prototype only needs three primitives:
//! - standard normal PDF
//! - standard normal CDF
//! - inverse standard normal CDF
//!
//! The approximations below are standard engineering approximations and are
//! more than accurate enough for codebook initialization and testing.

use std::f64::consts::{PI, SQRT_2};

pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Abramowitz and Stegun 7.1.26.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Acklam's inverse-normal approximation.
pub fn inverse_normal_cdf(p: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&p));
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let plow = 0.02425;
    let phigh = 1.0 - plow;

    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p > phigh {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

/// `E[X | a < X < b]` for `X ~ N(0, sigma^2)`.
pub fn gaussian_conditional_expectation(sigma: f64, a: f64, b: f64) -> f64 {
    let a_std = if a.is_finite() { a / sigma } else { a };
    let b_std = if b.is_finite() { b / sigma } else { b };

    let probability = if !a_std.is_finite() {
        normal_cdf(b_std)
    } else if !b_std.is_finite() {
        1.0 - normal_cdf(a_std)
    } else {
        normal_cdf(b_std) - normal_cdf(a_std)
    };

    if probability < 1e-15 {
        return if a.is_finite() && !b.is_finite() {
            a + sigma
        } else if !a.is_finite() && b.is_finite() {
            b - sigma
        } else if a.is_finite() && b.is_finite() {
            0.5 * (a + b)
        } else {
            0.0
        };
    }

    sigma * (normal_pdf(a_std) - normal_pdf(b_std)) / probability
}
