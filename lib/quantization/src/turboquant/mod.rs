//! A self-contained TurboQuant implementation that mirrors the high-level
//! structure used in `TheTom/turboquant_plus`, while staying ergonomic for
//! standalone Rust experiments.
//!
//! The implementation is intentionally split into small modules so we can
//! validate each stage independently:
//! - random / structured rotations
//! - Lloyd-Max level codebooks
//! - generic bit pack / unpack
//! - optional 1-bit QJL residuals
//! - optional norm-correction scaling
//! - plain and SIMD scoring paths
//!
//! This module does not integrate with Qdrant storage yet. It only focuses on
//! encoding, decoding, scoring, and recall evaluation so the algorithm can be
//! validated in isolation first.

mod codebook;
mod math;
mod packing;
mod qjl;
mod recall;
mod rotation;
mod simd;

use std::fmt;

pub use qjl::QjlResidual;
pub use recall::{
    ExactSearchBaseline, RecallAtK, RecallEvaluation, RecallReport, compute_exact_baseline,
    evaluate_recall, evaluate_recall_with_baseline,
};
pub use rotation::RotationKind;

use crate::EncodingError;

/// Runtime configuration for the standalone TurboQuant codec.
///
/// The four requested variants are all represented by this single config:
/// - baseline variant: `qjl = false`, `norm_correction = Disabled`
/// - QJL variant: `qjl = true`
/// - norm-correction variant: `norm_correction = Exact`
/// - SIMD variant: same encoding, but queried through `score_dot_simd`
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Original vector dimensionality.
    pub dim: usize,
    /// Number of first-stage quantization bits used by the level codec.
    pub bit_width: u8,
    /// Rotation backend used before level quantization.
    pub rotation: RotationKind,
    /// Seed used for random rotation / random projections.
    pub seed: u64,
    /// Whether a 1-bit QJL residual stage should be appended.
    pub qjl: bool,
    /// Norm-correction mode inspired by `spiritbuun/llama-cpp-turboquant-cuda`.
    pub norm_correction: NormCorrection,
}

impl TurboQuantConfig {
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

    /// Faithful baseline that reproduces the TheTom prototype shape:
    /// normalize -> rotate -> Lloyd-Max level quantize -> pack.
    pub fn baseline(dim: usize, bit_width: u8, seed: u64) -> Self {
        Self {
            dim,
            bit_width,
            rotation: RotationKind::Haar,
            seed,
            qjl: false,
            norm_correction: NormCorrection::Disabled,
        }
    }

    /// QJL variant that keeps the same baseline rotation path and adds a
    /// second 1-bit QJL residual.
    ///
    /// The TurboQuant paper only requires a random rotation. We keep
    /// `Hadamard` available as a structured backend, but default to the
    /// dense path here so the config matches the paper's simpler presentation.
    pub fn with_qjl(dim: usize, bit_width: u8, seed: u64) -> Self {
        Self {
            dim,
            bit_width,
            rotation: RotationKind::Haar,
            seed,
            qjl: true,
            norm_correction: NormCorrection::Disabled,
        }
    }

    /// 3-bit baseline path with norm correction enabled.
    pub fn with_norm_correction(dim: usize, bit_width: u8, seed: u64) -> Self {
        Self {
            dim,
            bit_width,
            rotation: RotationKind::Haar,
            seed,
            qjl: false,
            norm_correction: NormCorrection::Exact,
        }
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
    fn apply(self, original_norm: f32, reconstructed_rotated_norm: f32) -> f32 {
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

/// Encoded representation of one vector.
///
/// `packed_levels` always contains the level stage. `qjl` is only populated
/// when the codec was created with `config.qjl = true`.
#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantVector {
    pub packed_levels: Vec<u8>,
    pub scale: f32,
    pub qjl: Option<QjlResidual>,
}

impl TurboQuantVector {
    pub fn packed_len_bytes(dim: usize, bit_width: u8) -> usize {
        (dim * bit_width as usize).div_ceil(8)
    }
}

/// Standalone codec for the full TurboQuant family.
pub struct TurboQuantCodec {
    config: TurboQuantConfig,
    rotation: rotation::Rotation,
    codebook: codebook::QuantizationCodebook,
    qjl: Option<qjl::QjlProjector>,
}

impl fmt::Debug for TurboQuantCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TurboQuantCodec")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl TurboQuantCodec {
    pub fn new(config: TurboQuantConfig) -> Result<Self, EncodingError> {
        config.validate()?;

        let rotation = rotation::Rotation::new(config.rotation, config.dim, config.seed)?;
        let codebook = codebook::QuantizationCodebook::new(config.bit_width, config.dim);
        let qjl = config
            .qjl
            .then(|| qjl::QjlProjector::new(config.dim, config.seed ^ 0x5bf0_3635_d4f9_8a51));

        Ok(Self {
            config,
            rotation,
            codebook,
            qjl,
        })
    }

    pub fn config(&self) -> &TurboQuantConfig {
        &self.config
    }

    pub fn quantize(&self, vector: &[f32]) -> Result<TurboQuantVector, EncodingError> {
        if vector.len() != self.config.dim {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant expected dim {}, got {}",
                self.config.dim,
                vector.len()
            )));
        }

        let original_norm = l2_norm(vector);
        let safe_norm = if original_norm > 0.0 {
            original_norm
        } else {
            1.0
        };
        let normalized: Vec<f32> = vector.iter().map(|&v| v / safe_norm).collect();

        let rotated = self.rotation.apply(&normalized);

        let mut levels = Vec::with_capacity(self.config.dim);
        let mut rotated_reconstruction = Vec::with_capacity(self.config.dim);
        for &value in &rotated {
            let level = self.codebook.nearest_index(value);
            levels.push(level);
            rotated_reconstruction.push(self.codebook.level(level));
        }

        let scale = self
            .config
            .norm_correction
            .apply(original_norm, l2_norm(&rotated_reconstruction));

        let packed_levels = packing::pack_bits(&levels, self.config.bit_width);
        let qjl = self.qjl.as_ref().map(|projector| {
            let base_reconstruction = self
                .rotation
                .apply_transpose(&rotated_reconstruction, scale);
            let residual: Vec<f32> = vector
                .iter()
                .zip(&base_reconstruction)
                .map(|(&v, &hat)| v - hat)
                .collect();
            projector.quantize(&residual)
        });

        Ok(TurboQuantVector {
            packed_levels,
            scale,
            qjl,
        })
    }

    pub fn quantize_batch(
        &self,
        vectors: impl IntoIterator<Item = impl AsRef<[f32]>>,
    ) -> Result<Vec<TurboQuantVector>, EncodingError> {
        vectors
            .into_iter()
            .map(|vector| self.quantize(vector.as_ref()))
            .collect()
    }

    /// Dequantize into a fresh vector. This keeps the code easy to inspect.
    ///
    /// For performance-sensitive loops we prefer `dequantize_into`.
    pub fn dequantize(&self, encoded: &TurboQuantVector) -> Vec<f32> {
        let mut output = vec![0.0; self.config.dim];
        self.dequantize_into(encoded, &mut output);
        output
    }

    /// Dequantize into a caller-provided buffer to avoid repeated allocations
    /// during recall evaluation.
    pub fn dequantize_into(&self, encoded: &TurboQuantVector, output: &mut [f32]) {
        debug_assert_eq!(output.len(), self.config.dim);

        let levels = packing::unpack_bits(
            &encoded.packed_levels,
            self.config.bit_width,
            self.config.dim,
        );
        let rotated_reconstruction: Vec<f32> = levels
            .iter()
            .map(|&level| self.codebook.level(level))
            .collect();

        let mut base = self
            .rotation
            .apply_transpose(&rotated_reconstruction, encoded.scale);

        if let (Some(projector), Some(qjl)) = (&self.qjl, &encoded.qjl) {
            let residual = projector.dequantize(qjl);
            for (value, residual_value) in base.iter_mut().zip(residual) {
                *value += residual_value;
            }
        }

        output.copy_from_slice(&base);
    }

    /// Plain score path used for correctness baselines.
    pub fn score_dot_plain(&self, query: &[f32], encoded: &TurboQuantVector) -> f32 {
        let reconstruction = self.dequantize(encoded);
        query
            .iter()
            .zip(&reconstruction)
            .map(|(&a, &b)| a * b)
            .sum()
    }

    /// SIMD-capable score path. The encoding is identical to the plain path;
    /// only the final dot product is accelerated.
    pub fn score_dot_simd(&self, query: &[f32], encoded: &TurboQuantVector) -> f32 {
        let reconstruction = self.dequantize(encoded);
        simd::dot(query, &reconstruction)
    }
}

pub(crate) fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|&v| v * v).sum::<f32>().sqrt()
}
