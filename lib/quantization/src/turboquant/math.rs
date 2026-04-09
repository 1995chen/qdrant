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

use rand::RngExt;

pub fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

pub fn sample_standard_normal_pair(rng: &mut impl RngExt) -> (f32, f32) {
    let u1 = (1.0f32 - rng.random::<f32>()).max(1e-12f32);
    let u2 = rng.random::<f32>();
    let radius = (-2.0f32 * u1.ln()).sqrt();
    let theta = 2.0f32 * std::f32::consts::PI * u2;
    (radius * theta.cos(), radius * theta.sin())
}

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

#[cfg(test)]
mod tests {
    use super::{gaussian_conditional_expectation, inverse_normal_cdf, normal_cdf};

    #[test]
    fn normal_cdf_is_symmetric_and_monotonic() {
        let xs = [-6.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0];

        for pair in xs.windows(2) {
            assert!(
                normal_cdf(pair[0]) < normal_cdf(pair[1]),
                "normal_cdf should be strictly increasing: x0={}, x1={}",
                pair[0],
                pair[1]
            );
        }

        let tol = 2e-7;
        for x in xs {
            let lhs = normal_cdf(-x);
            let rhs = 1.0 - normal_cdf(x);
            assert!(
                (lhs - rhs).abs() <= tol,
                "symmetry mismatch at x={x}: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    #[test]
    fn inverse_normal_cdf_is_monotonic_and_roundtrips_cdf() {
        let ps = [
            1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99,
        ];

        for pair in ps.windows(2) {
            assert!(
                inverse_normal_cdf(pair[0]) < inverse_normal_cdf(pair[1]),
                "inverse_normal_cdf should be strictly increasing: p0={}, p1={}",
                pair[0],
                pair[1]
            );
        }

        let tol = 2e-7;
        for p in ps {
            let reconstructed = normal_cdf(inverse_normal_cdf(p));
            assert!(
                (reconstructed - p).abs() <= tol,
                "roundtrip mismatch at p={p}: reconstructed={reconstructed}"
            );
        }
    }

    #[test]
    fn inverse_normal_cdf_matches_known_quantiles_and_endpoints() {
        let known_quantiles = [
            (0.5, 0.0),
            (0.15865525393145707, -1.0),
            (0.8413447460685429, 1.0),
            (0.022750131948179195, -2.0),
            (0.9772498680518208, 2.0),
            (0.0013498980316300933, -3.0),
            (0.9986501019683699, 3.0),
        ];

        let tol = 5e-7;
        for (p, expected_z) in known_quantiles {
            let actual_z = inverse_normal_cdf(p);
            assert!(
                (actual_z - expected_z).abs() <= tol,
                "quantile mismatch at p={p}: expected {expected_z}, got {actual_z}"
            );
        }

        assert_eq!(inverse_normal_cdf(0.0), f64::NEG_INFINITY);
        assert_eq!(inverse_normal_cdf(1.0), f64::INFINITY);
    }

    #[test]
    fn gaussian_conditional_expectation_stays_within_bounds() {
        let finite_cases = [
            (1.0, -1.0, 1.0),
            (0.3, -0.2, 0.4),
            (0.2, -0.5, -0.1),
            (0.4, 0.1, 0.6),
        ];

        for (sigma, a, b) in finite_cases {
            let value = gaussian_conditional_expectation(sigma, a, b);
            assert!(
                value >= a && value <= b,
                "expected value to stay in [{a}, {b}], got {value}"
            );
        }

        let lower_tail = gaussian_conditional_expectation(0.3, f64::NEG_INFINITY, 0.2);
        assert!(
            lower_tail <= 0.2,
            "expected lower-tail conditional mean <= upper bound, got {lower_tail}"
        );

        let upper_tail = gaussian_conditional_expectation(0.3, -0.2, f64::INFINITY);
        assert!(
            upper_tail >= -0.2,
            "expected upper-tail conditional mean >= lower bound, got {upper_tail}"
        );
    }

    #[test]
    fn gaussian_conditional_expectation_is_zero_on_symmetric_intervals() {
        let intervals = [(-3.0, 3.0), (-1.0, 1.0), (-0.5, 0.5)];
        let sigmas = [0.1, 0.3, 1.0];

        let tol = 1e-10;
        for sigma in sigmas {
            for (a_std, b_std) in intervals {
                let a = a_std * sigma;
                let b = b_std * sigma;
                let value = gaussian_conditional_expectation(sigma, a, b);
                assert!(
                    value.abs() <= tol,
                    "expected near-zero mean on symmetric interval [{a}, {b}], got {value}"
                );
            }
        }
    }

    #[test]
    fn gaussian_conditional_expectation_is_zero_for_full_distribution() {
        let sigmas = [0.1, 0.3, 1.0, 10.0];

        for sigma in sigmas {
            let value = gaussian_conditional_expectation(sigma, f64::NEG_INFINITY, f64::INFINITY);
            assert_eq!(
                value, 0.0,
                "expected zero mean for sigma={sigma}, got {value}"
            );
        }
    }

    #[test]
    fn gaussian_conditional_expectation_uses_stable_fallbacks_for_tiny_probabilities() {
        let sigma = 1.0;

        let degenerate = gaussian_conditional_expectation(sigma, 2.5, 2.5);
        assert_eq!(degenerate, 2.5);

        let tiny_interval = gaussian_conditional_expectation(sigma, 10.0, 10.0 + 1e-16);
        assert_eq!(
            tiny_interval, 10.0,
            "expected midpoint fallback for tiny interval"
        );

        let upper_tail = gaussian_conditional_expectation(sigma, 10.0, f64::INFINITY);
        assert_eq!(
            upper_tail, 11.0,
            "expected upper-tail fallback for tiny mass"
        );

        let lower_tail = gaussian_conditional_expectation(sigma, f64::NEG_INFINITY, -10.0);
        assert_eq!(
            lower_tail, -11.0,
            "expected lower-tail fallback for tiny mass"
        );
    }
}
