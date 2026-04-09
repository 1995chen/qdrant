//! Lloyd-Max quantization codebook construction.
//!
//! This is a direct Rust translation of the TheTom prototype logic:
//! - 1-bit: closed-form centroid
//! - 2-bit: fixed centroid set from the paper / prototype
//! - 3+ bits: Lloyd iterations on a Gaussian approximation

use super::math::{gaussian_conditional_expectation, inverse_normal_cdf};

#[derive(Clone, Debug)]
pub struct QuantizationCodebook {
    levels: Vec<f32>,
    boundaries: Vec<f32>,
}

impl QuantizationCodebook {
    pub fn new(bit_width: u8, dim: usize) -> Self {
        let levels = Self::optimal_centroids(bit_width, dim);
        let boundaries = levels
            .windows(2)
            .map(|pair| 0.5 * (pair[0] + pair[1]))
            .collect();
        Self { levels, boundaries }
    }

    pub fn level(&self, index: u8) -> f32 {
        self.levels[index as usize]
    }

    pub fn nearest_index(&self, value: f32) -> u8 {
        let boundary_index = self
            .boundaries
            .partition_point(|boundary| *boundary < value);
        boundary_index as u8
    }

    fn optimal_centroids(bit_width: u8, dim: usize) -> Vec<f32> {
        let sigma = 1.0 / (dim as f64).sqrt();
        match bit_width {
            1 => {
                let c = (2.0 / (std::f64::consts::PI * dim as f64)).sqrt() as f32;
                vec![-c, c]
            }
            2 => vec![-1.51, -0.453, 0.453, 1.51]
                .into_iter()
                .map(|value| (value / (dim as f64).sqrt()) as f32)
                .collect(),
            _ => Self::lloyds_gaussian(1usize << bit_width, sigma),
        }
    }

    fn lloyds_gaussian(level_count: usize, sigma: f64) -> Vec<f32> {
        let mut boundaries: Vec<f64> = (1..level_count)
            .map(|i| inverse_normal_cdf(i as f64 / level_count as f64) * sigma)
            .collect();

        let mut centroids = vec![0.0f64; level_count];
        Self::update_centroids(&mut centroids, &boundaries, sigma);

        for _ in 0..100 {
            for (boundary, pair) in boundaries.iter_mut().zip(centroids.windows(2)) {
                *boundary = 0.5 * (pair[0] + pair[1]);
            }
            Self::update_centroids(&mut centroids, &boundaries, sigma);
        }

        centroids.into_iter().map(|value| value as f32).collect()
    }

    fn update_centroids(centroids: &mut [f64], boundaries: &[f64], sigma: f64) {
        if centroids.is_empty() {
            return;
        }

        centroids[0] = gaussian_conditional_expectation(sigma, f64::NEG_INFINITY, boundaries[0]);
        for index in 1..centroids.len().saturating_sub(1) {
            centroids[index] =
                gaussian_conditional_expectation(sigma, boundaries[index - 1], boundaries[index]);
        }
        centroids[centroids.len() - 1] = gaussian_conditional_expectation(
            sigma,
            boundaries[boundaries.len() - 1],
            f64::INFINITY,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::QuantizationCodebook;

    #[test]
    fn codebook_has_expected_number_of_levels_and_boundaries() {
        for bit_width in [1_u8, 2, 3, 4] {
            let codebook = QuantizationCodebook::new(bit_width, 64);
            let level_count = 1usize << bit_width;
            assert_eq!(codebook.levels.len(), level_count);
            assert_eq!(codebook.boundaries.len(), level_count - 1);
        }
    }

    #[test]
    fn codebook_levels_and_boundaries_are_strictly_increasing() {
        let codebook = QuantizationCodebook::new(4, 64);

        for pair in codebook.levels.windows(2) {
            assert!(
                pair[0] < pair[1],
                "levels must be strictly increasing: {:?}",
                codebook.levels
            );
        }

        for pair in codebook.boundaries.windows(2) {
            assert!(
                pair[0] < pair[1],
                "boundaries must be strictly increasing: {:?}",
                codebook.boundaries
            );
        }
    }

    #[test]
    fn one_bit_codebook_matches_closed_form_centroids() {
        let dim = 64usize;
        let codebook = QuantizationCodebook::new(1, dim);
        let expected = (2.0 / (std::f64::consts::PI * dim as f64)).sqrt() as f32;

        assert!((codebook.level(0) + expected).abs() < 1e-7);
        assert!((codebook.level(1) - expected).abs() < 1e-7);
        assert!(codebook.boundaries[0].abs() < 1e-7);
    }

    #[test]
    fn nearest_index_respects_interval_boundaries() {
        let codebook = QuantizationCodebook::new(3, 64);

        for index in 0..codebook.levels.len() {
            let level = codebook.level(index as u8);
            assert_eq!(codebook.nearest_index(level), index as u8);
        }

        for (index, &boundary) in codebook.boundaries.iter().enumerate() {
            assert_eq!(codebook.nearest_index(boundary - 1e-6), index as u8);
            assert_eq!(codebook.nearest_index(boundary), index as u8);
            assert_eq!(codebook.nearest_index(boundary + 1e-6), index as u8 + 1);
        }
    }

    #[test]
    fn two_bit_codebook_is_symmetric() {
        let codebook = QuantizationCodebook::new(2, 64);
        let last = codebook.levels.len() - 1;

        for index in 0..codebook.levels.len() {
            let mirrored = codebook.levels[last - index];
            assert!(
                (codebook.levels[index] + mirrored).abs() < 1e-6,
                "expected symmetric levels, got {:?}",
                codebook.levels
            );
        }
    }
}
