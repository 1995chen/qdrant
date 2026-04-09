//! Rotation backends.
//!
//! We keep two variants:
//! - `Haar`: the default dense random rotation used to match the paper's
//!   simple "random rotation" description.
//! - `Hadamard`: optional structured rotation with random sign diagonals.
//!   This is a practical implementation choice, not a requirement of the core
//!   TurboQuant algorithm.

mod haar;
mod hadamard;

use self::haar::HaarRotation;
use self::hadamard::HadamardRotation;
use crate::EncodingError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationKind {
    Haar,
    Hadamard,
}

#[derive(Clone)]
pub enum Rotation {
    Haar(HaarRotation),
    Hadamard(HadamardRotation),
}

impl Rotation {
    pub fn new(kind: RotationKind, dim: usize, seed: u64) -> Result<Self, EncodingError> {
        Ok(match kind {
            RotationKind::Haar => Self::Haar(HaarRotation::new(dim, seed)),
            RotationKind::Hadamard => Self::Hadamard(HadamardRotation::new(dim, seed)),
        })
    }

    pub fn apply(&self, input: &[f32]) -> Vec<f32> {
        match self {
            Self::Haar(rotation) => rotation.apply(input),
            Self::Hadamard(rotation) => rotation.apply(input),
        }
    }

    pub fn apply_transpose(&self, input: &[f32], scale: f32) -> Vec<f32> {
        match self {
            Self::Haar(rotation) => rotation.apply_transpose(input, scale),
            Self::Hadamard(rotation) => rotation.apply_transpose(input, scale),
        }
    }
}
