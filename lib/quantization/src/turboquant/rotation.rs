//! Rotation backends.
//!
//! We keep two variants:
//! - `DenseHaar`: closest to the dense random rotation in the reference
//!   Python prototype.
//! - `WalshHadamard`: structured rotation with random sign diagonals. This is
//!   the practical path discussed heavily in `llama.cpp` discussion #20969.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::EncodingError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationKind {
    DenseHaar,
    WalshHadamard,
}

#[derive(Clone)]
pub enum Rotation {
    Dense(DenseRotation),
    WalshHadamard(WalshHadamardRotation),
}

impl Rotation {
    pub fn new(kind: RotationKind, dim: usize, seed: u64) -> Result<Self, EncodingError> {
        Ok(match kind {
            RotationKind::DenseHaar => Self::Dense(DenseRotation::new(dim, seed)),
            RotationKind::WalshHadamard => {
                Self::WalshHadamard(WalshHadamardRotation::new(dim, seed))
            }
        })
    }

    pub fn apply(&self, input: &[f32]) -> Vec<f32> {
        match self {
            Self::Dense(rotation) => rotation.apply(input),
            Self::WalshHadamard(rotation) => rotation.apply(input),
        }
    }

    pub fn apply_transpose(&self, input: &[f32], scale: f32) -> Vec<f32> {
        match self {
            Self::Dense(rotation) => rotation.apply_transpose(input, scale),
            Self::WalshHadamard(rotation) => rotation.apply_transpose(input, scale),
        }
    }
}

#[derive(Clone)]
pub struct DenseRotation {
    dim: usize,
    matrix: Vec<f32>,
}

impl DenseRotation {
    fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut matrix = vec![0.0f32; dim * dim];
        let mut filled = 0usize;
        while filled < matrix.len() {
            let (z0, z1) = box_muller(&mut rng);
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
                let dot = dot(
                    &matrix[row_index * dim..(row_index + 1) * dim],
                    &matrix[previous_row * dim..(previous_row + 1) * dim],
                );
                for column in 0..dim {
                    matrix[row_index * dim + column] -= dot * matrix[previous_row * dim + column];
                }
            }

            let row = &mut matrix[row_index * dim..(row_index + 1) * dim];
            let norm = dot(row, row).sqrt();
            if norm <= 1e-12 {
                row.fill(0.0);
                row[row_index] = 1.0;
            } else {
                row.iter_mut().for_each(|value| *value /= norm);
            }
        }

        Self { dim, matrix }
    }

    fn apply(&self, input: &[f32]) -> Vec<f32> {
        self.matrix
            .chunks_exact(self.dim)
            .map(|row| dot(row, input))
            .collect()
    }

    fn apply_transpose(&self, input: &[f32], scale: f32) -> Vec<f32> {
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

#[derive(Clone)]
pub struct WalshHadamardRotation {
    dim: usize,
    padded_dim: usize,
    left_signs: Vec<f32>,
    right_signs: Vec<f32>,
}

impl WalshHadamardRotation {
    fn new(dim: usize, seed: u64) -> Self {
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

    fn apply(&self, input: &[f32]) -> Vec<f32> {
        let mut scratch = vec![0.0f32; self.padded_dim];
        for index in 0..self.dim {
            scratch[index] = input[index] * self.right_signs[index];
        }
        fwht(&mut scratch);
        let scale = 1.0 / (self.padded_dim as f32).sqrt();
        scratch
            .iter()
            .zip(&self.left_signs)
            .take(self.dim)
            .map(|(&value, &sign)| value * sign * scale)
            .collect()
    }

    fn apply_transpose(&self, input: &[f32], scale: f32) -> Vec<f32> {
        let mut scratch = vec![0.0f32; self.padded_dim];
        for index in 0..self.dim {
            scratch[index] = input[index] * self.left_signs[index];
        }
        fwht(&mut scratch);
        let inverse_scale = scale / (self.padded_dim as f32).sqrt();
        scratch
            .iter()
            .zip(&self.right_signs)
            .take(self.dim)
            .map(|(&value, &sign)| value * sign * inverse_scale)
            .collect()
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
