//! 1-bit Quantized Johnson-Lindenstrauss residual stage.
//!
//! This follows the prototype semantics closely:
//! - store one sign bit per projected residual dimension
//! - store the residual L2 norm as a scalar
//! - reconstruct with `sqrt(pi / 2) / d * S^T * sign * norm`
//!
//! The implementation intentionally keeps the full dense projection matrix.
//! That is the simplest faithful standalone version for algorithm validation.

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::{math, packing};
use crate::EncodingError;

const QJL_CONST: f32 = 1.253_314_1;

#[derive(Clone, Debug, PartialEq)]
pub struct QjlResidual {
    packed_signs: Vec<u8>,
    norm: f32,
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
            let (z0, z1) = math::sample_standard_normal_pair(&mut rng);
            matrix.push(z0);
            if matrix.len() < dim * dim {
                matrix.push(z1);
            }
        }
        Self { dim, matrix }
    }

    pub fn quantize(&self, residual: &[f32]) -> QjlResidual {
        let norm = math::l2_norm(residual);
        let mut signs = Vec::with_capacity(self.dim);

        for row in self.matrix.chunks_exact(self.dim) {
            let projection = row.iter().zip(residual).map(|(&a, &b)| a * b).sum::<f32>();
            signs.push(u8::from(projection >= 0.0));
        }

        QjlResidual::new(packing::pack_bits(&signs, 1), norm)
    }

    #[allow(dead_code)]
    pub fn dequantize(&self, residual: &QjlResidual) -> Result<Vec<f32>, EncodingError> {
        let mut output = vec![0.0; self.dim];
        self.dequantize_into(residual, &mut output)?;
        Ok(output)
    }

    pub fn dequantize_into(
        &self,
        residual: &QjlResidual,
        output: &mut [f32],
    ) -> Result<(), EncodingError> {
        residual.validate(self.dim)?;
        if output.len() != self.dim {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant QJL output expected dim {}, got {}",
                self.dim,
                output.len()
            )));
        }

        if residual.norm() == 0.0 {
            output.fill(0.0);
            return Ok(());
        }

        output.fill(0.0);
        for row in 0..self.dim {
            let byte = residual.packed_signs()[row / 8];
            let bit = (byte >> (row % 8)) & 1;
            let sign = if bit == 0 { -1.0 } else { 1.0 };
            let row_slice = &self.matrix[row * self.dim..(row + 1) * self.dim];
            for (out, &value) in output.iter_mut().zip(row_slice) {
                *out += sign * value;
            }
        }

        let scale = QJL_CONST * residual.norm() / self.dim as f32;
        output.iter_mut().for_each(|value| *value *= scale);
        Ok(())
    }
}

impl QjlResidual {
    pub(crate) fn new(packed_signs: Vec<u8>, norm: f32) -> Self {
        Self { packed_signs, norm }
    }

    pub fn packed_signs(&self) -> &[u8] {
        &self.packed_signs
    }

    pub fn norm(&self) -> f32 {
        self.norm
    }

    pub(crate) fn validate(&self, dim: usize) -> Result<(), EncodingError> {
        let expected_bytes = dim.div_ceil(8);
        if self.packed_signs.len() != expected_bytes {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant QJL residual expected {expected_bytes} bytes, got {}",
                self.packed_signs.len()
            )));
        }
        if self.norm.is_sign_negative() {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant QJL residual norm must be non-negative, got {}",
                self.norm
            )));
        }
        Ok(())
    }
}
