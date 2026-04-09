//! 1-bit Quantized Johnson-Lindenstrauss residual stage.
//!
//! This follows the prototype semantics closely:
//! - store one sign bit per projected residual dimension
//! - store the residual L2 norm as a scalar
//! - reconstruct with `sqrt(pi / 2) / d * S^T * sign * norm`
//!
//! The implementation intentionally keeps the full dense projection matrix.
//! That is the simplest faithful standalone version for algorithm validation.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::{l2_norm, packing};

const QJL_CONST: f32 = 1.253_314_1;

#[derive(Clone, Debug, PartialEq)]
pub struct QjlResidual {
    pub packed_signs: Vec<u8>,
    pub norm: f32,
}

#[derive(Clone, Debug)]
pub struct QjlProjector {
    dim: usize,
    matrix: Vec<f32>,
}

impl QjlProjector {
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut matrix = Vec::with_capacity(dim * dim);
        while matrix.len() < dim * dim {
            let (z0, z1) = Self::box_muller(&mut rng);
            matrix.push(z0);
            if matrix.len() < dim * dim {
                matrix.push(z1);
            }
        }
        Self { dim, matrix }
    }

    pub fn quantize(&self, residual: &[f32]) -> QjlResidual {
        let norm = l2_norm(residual);
        let mut signs = Vec::with_capacity(self.dim);

        for row in self.matrix.chunks_exact(self.dim) {
            let projection = row.iter().zip(residual).map(|(&a, &b)| a * b).sum::<f32>();
            signs.push(u8::from(projection >= 0.0));
        }

        QjlResidual {
            packed_signs: packing::pack_bits(&signs, 1),
            norm,
        }
    }

    pub fn dequantize(&self, residual: &QjlResidual) -> Vec<f32> {
        if residual.norm == 0.0 {
            return vec![0.0; self.dim];
        }

        let signs = packing::unpack_bits(&residual.packed_signs, 1, self.dim);
        let mut reconstruction = vec![0.0f32; self.dim];

        for row in 0..self.dim {
            let sign = if signs[row] == 0 { -1.0 } else { 1.0 };
            let row_slice = &self.matrix[row * self.dim..(row + 1) * self.dim];
            for (out, &value) in reconstruction.iter_mut().zip(row_slice) {
                *out += sign * value;
            }
        }

        let scale = QJL_CONST * residual.norm / self.dim as f32;
        reconstruction.iter_mut().for_each(|value| *value *= scale);
        reconstruction
    }

    fn box_muller(rng: &mut StdRng) -> (f32, f32) {
        let u1 = (1.0f32 - rng.random::<f32>()).max(1e-12f32);
        let u2 = rng.random::<f32>();
        let radius = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * std::f32::consts::PI * u2;
        (radius * theta.cos(), radius * theta.sin())
    }
}
