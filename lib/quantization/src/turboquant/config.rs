use super::rotation::RotationKind;
use crate::EncodingError;

/// Runtime configuration for the standalone TurboQuant codec.
///
/// The different TurboQuant variants are represented by this single config.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Original vector dimensionality.
    dim: usize,
    /// Number of first-stage quantization bits used by the level codec.
    bit_width: u8,
    /// Rotation backend used before level quantization.
    rotation: RotationKind,
    /// Seed used for random rotation / random projections.
    seed: u64,
    /// Whether a 1-bit QJL residual stage should be appended.
    qjl: bool,
    /// Norm-correction mode inspired by `spiritbuun/llama-cpp-turboquant-cuda`.
    norm_correction: NormCorrection,
}

impl TurboQuantConfig {
    /// Baseline configuration:
    /// normalize -> rotate -> Lloyd-Max level quantize -> pack.
    #[must_use]
    pub fn new(dim: usize, bit_width: u8, seed: u64) -> Self {
        Self {
            dim,
            bit_width,
            rotation: RotationKind::Haar,
            seed,
            qjl: false,
            norm_correction: NormCorrection::Disabled,
        }
    }

    #[must_use]
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    #[must_use]
    pub fn with_bit_width(mut self, bit_width: u8) -> Self {
        self.bit_width = bit_width;
        self
    }

    #[must_use]
    pub fn with_rotation(mut self, rotation: RotationKind) -> Self {
        self.rotation = rotation;
        self
    }

    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    #[must_use]
    pub fn with_qjl(mut self, qjl: bool) -> Self {
        self.qjl = qjl;
        self
    }

    #[must_use]
    pub fn with_norm_correction(mut self, norm_correction: NormCorrection) -> Self {
        self.norm_correction = norm_correction;
        self
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    pub fn rotation(&self) -> RotationKind {
        self.rotation
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn qjl(&self) -> bool {
        self.qjl
    }

    pub fn norm_correction(&self) -> NormCorrection {
        self.norm_correction
    }

    pub fn validate(&self) -> Result<(), EncodingError> {
        if self.dim == 0 {
            return Err(EncodingError::ArgumentsError(
                "TurboQuant dimension must be non-zero".to_owned(),
            ));
        }
        if self.bit_width == 0 || self.bit_width > 8 {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant bit width must be in 1..=8, got {}",
                self.bit_width
            )));
        }
        Ok(())
    }
}

/// Norm-correction policy.
///
/// In the Python prototype, correction is implemented by re-normalizing the
/// reconstructed rotated vector before the inverse rotation. The CUDA branch
/// described in the discussion stores an equivalent multiplicative scale
/// `original_norm / reconstructed_norm`, which lets decode stay cheap.
///
/// We use the stored-scale form here because it is a better fit for packed
/// vector storage and avoids an extra norm computation during recall tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormCorrection {
    Disabled,
    Exact,
    /// Interpolates between no correction (`strength = 0`) and the exact scale
    /// (`strength = 1`). This is useful when we want to mimic the "tunable"
    /// correction experiments mentioned in the discussion.
    Interpolated {
        strength: f32,
    },
}

impl NormCorrection {
    pub(crate) fn apply(self, original_norm: f32, reconstructed_rotated_norm: f32) -> f32 {
        if original_norm == 0.0 {
            return 0.0;
        }

        let safe_recon_norm = if reconstructed_rotated_norm > 1e-12 {
            reconstructed_rotated_norm
        } else {
            1.0
        };

        match self {
            Self::Disabled => original_norm,
            Self::Exact => original_norm / safe_recon_norm,
            Self::Interpolated { strength } => {
                let exact = 1.0 / safe_recon_norm;
                let blended = 1.0 + strength.clamp(0.0, 1.0) * (exact - 1.0);
                original_norm * blended
            }
        }
    }
}
