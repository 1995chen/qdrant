use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[derive(Clone)]
pub(crate) struct HarrRotation {
    dim: usize,
    matrix: Vec<f32>,
}

impl HarrRotation {
    pub(super) fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut matrix = vec![0.0f32; dim * dim];
        let mut filled = 0usize;
        while filled < matrix.len() {
            let (z0, z1) = Self::box_muller(&mut rng);
            matrix[filled] = z0;
            filled += 1;
            if filled < matrix.len() {
                matrix[filled] = z1;
                filled += 1;
            }
        }

        // Modified Gram-Schmidt produces an orthonormal row basis. That gives
        // us a genuine rotation-like orthogonal matrix without external linear
        // algebra dependencies.
        for row_index in 0..dim {
            for previous_row in 0..row_index {
                let dot = Self::dot(
                    &matrix[row_index * dim..(row_index + 1) * dim],
                    &matrix[previous_row * dim..(previous_row + 1) * dim],
                );
                for column in 0..dim {
                    matrix[row_index * dim + column] -= dot * matrix[previous_row * dim + column];
                }
            }

            let row = &mut matrix[row_index * dim..(row_index + 1) * dim];
            let norm = Self::dot(row, row).sqrt();
            if norm <= 1e-12 {
                row.fill(0.0);
                row[row_index] = 1.0;
            } else {
                row.iter_mut().for_each(|value| *value /= norm);
            }
        }

        Self { dim, matrix }
    }

    fn box_muller(rng: &mut StdRng) -> (f32, f32) {
        let u1 = (1.0f32 - rng.random::<f32>()).max(1e-12f32);
        let u2 = rng.random::<f32>();
        let radius = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * std::f32::consts::PI * u2;
        (radius * theta.cos(), radius * theta.sin())
    }

    fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter().zip(rhs).map(|(&a, &b)| a * b).sum()
    }

    pub(super) fn apply(&self, input: &[f32]) -> Vec<f32> {
        self.matrix
            .chunks_exact(self.dim)
            .map(|row| Self::dot(row, input))
            .collect()
    }

    pub(super) fn apply_transpose(&self, input: &[f32], scale: f32) -> Vec<f32> {
        let mut output = vec![0.0f32; self.dim];
        for row in 0..self.dim {
            let coefficient = input[row] * scale;
            let row_slice = &self.matrix[row * self.dim..(row + 1) * self.dim];
            for (out, &value) in output.iter_mut().zip(row_slice) {
                *out += coefficient * value;
            }
        }
        output
    }
}
