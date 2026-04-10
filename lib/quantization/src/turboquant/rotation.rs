//! Rotation backends.
//!
//! We currently keep a single variant:
//! - `Haar`: the default dense random rotation used to match the paper's
//!   simple "random rotation" description.

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::EncodingError;
use crate::turboquant::math;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationKind {
    Haar,
}

#[derive(Clone)]
pub enum Rotation {
    Haar(HaarRotation),
}

impl Rotation {
    pub fn new(kind: RotationKind, dim: usize, seed: u64) -> Result<Self, EncodingError> {
        Ok(match kind {
            RotationKind::Haar => Self::Haar(HaarRotation::new(dim, seed)),
        })
    }

    pub fn apply(&self, input: &[f32]) -> Vec<f32> {
        match self {
            Self::Haar(rotation) => rotation.apply(input),
        }
    }

    pub fn apply_transpose(&self, input: &[f32], scale: f32) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        self.apply_transpose_into(input, scale, &mut output);
        output
    }

    pub fn apply_transpose_into(&self, input: &[f32], scale: f32, output: &mut [f32]) {
        match self {
            Self::Haar(rotation) => rotation.apply_transpose_into(input, scale, output),
        }
    }
}

#[derive(Clone)]
pub(crate) struct HaarRotation {
    dim: usize,
    matrix: Vec<f32>,
}

impl HaarRotation {
    fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut matrix = vec![0.0f32; dim * dim];
        let mut filled = 0usize;
        while filled < matrix.len() {
            let (z0, z1) = math::sample_standard_normal_pair(&mut rng);
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

    fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
        lhs.iter().zip(rhs).map(|(&a, &b)| a * b).sum()
    }

    fn apply(&self, input: &[f32]) -> Vec<f32> {
        self.matrix
            .chunks_exact(self.dim)
            .map(|row| Self::dot(row, input))
            .collect()
    }

    fn apply_transpose_into(&self, input: &[f32], scale: f32, output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.dim);
        debug_assert_eq!(output.len(), self.dim);
        output.fill(0.0);
        for row in 0..self.dim {
            let coefficient = input[row] * scale;
            let row_slice = &self.matrix[row * self.dim..(row + 1) * self.dim];
            for (out, &value) in output.iter_mut().zip(row_slice) {
                *out += coefficient * value;
            }
        }
    }
}
