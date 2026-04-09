use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[derive(Clone)]
pub(crate) struct HadamardRotation {
    dim: usize,
    padded_dim: usize,
    left_signs: Vec<f32>,
    right_signs: Vec<f32>,
}

impl HadamardRotation {
    pub(crate) fn new(dim: usize, seed: u64) -> Self {
        let padded_dim = dim.next_power_of_two();
        let mut rng = StdRng::seed_from_u64(seed);
        let left_signs = (0..padded_dim)
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let right_signs = (0..padded_dim)
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect();
        Self {
            dim,
            padded_dim,
            left_signs,
            right_signs,
        }
    }

    fn fwht(values: &mut [f32]) {
        let mut width = 1;
        while width < values.len() {
            let step = width * 2;
            for chunk in values.chunks_exact_mut(step) {
                let (left, right) = chunk.split_at_mut(width);
                for index in 0..width {
                    let a = left[index];
                    let b = right[index];
                    left[index] = a + b;
                    right[index] = a - b;
                }
            }
            width = step;
        }
    }

    pub(crate) fn apply(&self, input: &[f32]) -> Vec<f32> {
        let mut scratch = vec![0.0f32; self.padded_dim];
        for index in 0..self.dim {
            scratch[index] = input[index] * self.right_signs[index];
        }
        Self::fwht(&mut scratch);
        let scale = 1.0 / (self.padded_dim as f32).sqrt();
        scratch
            .iter()
            .zip(&self.left_signs)
            .take(self.dim)
            .map(|(&value, &sign)| value * sign * scale)
            .collect()
    }

    pub(crate) fn apply_transpose(&self, input: &[f32], scale: f32) -> Vec<f32> {
        let mut scratch = vec![0.0f32; self.padded_dim];
        for index in 0..self.dim {
            scratch[index] = input[index] * self.left_signs[index];
        }
        Self::fwht(&mut scratch);
        let inverse_scale = scale / (self.padded_dim as f32).sqrt();
        scratch
            .iter()
            .zip(&self.right_signs)
            .take(self.dim)
            .map(|(&value, &sign)| value * sign * inverse_scale)
            .collect()
    }
}
