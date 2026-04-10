use super::{TurboQuantConfig, TurboQuantVector, codebook, math, packing, qjl, rotation, simd};
use crate::EncodingError;

/// Standalone codec for the full TurboQuant family.
pub struct TurboQuantCodec {
    config: TurboQuantConfig,
    rotation: rotation::Rotation,
    codebook: codebook::QuantizationCodebook,
    qjl: Option<qjl::QjlProjector>,
}

impl TurboQuantCodec {
    pub fn new(config: TurboQuantConfig) -> Result<Self, EncodingError> {
        config.validate()?;

        let rotation = rotation::Rotation::new(config.rotation(), config.dim(), config.seed())?;
        let codebook = codebook::QuantizationCodebook::new(config.bit_width(), config.dim());
        let qjl = config
            .qjl()
            .then(|| qjl::QjlProjector::new(config.dim(), config.seed() ^ 0x5bf0_3635_d4f9_8a51));

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
        if vector.len() != self.config.dim() {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant expected dim {}, got {}",
                self.config.dim(),
                vector.len()
            )));
        }

        let original_norm = math::l2_norm(vector);
        let safe_norm = if original_norm > 0.0 {
            original_norm
        } else {
            1.0
        };
        let normalized: Vec<f32> = vector.iter().map(|&v| v / safe_norm).collect();

        let rotated = self.rotation.apply(&normalized);

        let mut levels = Vec::with_capacity(self.config.dim());
        let mut rotated_reconstruction = Vec::with_capacity(self.config.dim());
        for &value in &rotated {
            let level = self.codebook.nearest_index(value);
            levels.push(level);
            rotated_reconstruction.push(self.codebook.level(level));
        }

        let scale = self
            .config
            .norm_correction()
            .apply(original_norm, math::l2_norm(&rotated_reconstruction));

        let packed_levels = packing::pack_bits(&levels, self.config.bit_width());
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

        Ok(TurboQuantVector::new(packed_levels, scale, qjl))
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
    pub fn dequantize(&self, encoded: &TurboQuantVector) -> Result<Vec<f32>, EncodingError> {
        let mut output = vec![0.0; self.config.dim()];
        self.dequantize_into(encoded, &mut output)?;
        Ok(output)
    }

    /// Dequantize into a caller-provided buffer to avoid repeated allocations
    /// during recall evaluation.
    pub fn dequantize_into(
        &self,
        encoded: &TurboQuantVector,
        output: &mut [f32],
    ) -> Result<(), EncodingError> {
        if output.len() != self.config.dim() {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant dequantize output expected dim {}, got {}",
                self.config.dim(),
                output.len()
            )));
        }

        let mut rotated_reconstruction = vec![0.0f32; self.config.dim()];
        let bit_width = self.config.bit_width() as usize;
        let mask = if bit_width == 8 {
            u16::MAX
        } else {
            (1u16 << bit_width) - 1
        };
        let mut bit_offset = 0usize;
        for rotated_value in &mut rotated_reconstruction {
            let byte_index = bit_offset / 8;
            let shift = bit_offset % 8;
            let mut level = (encoded.packed_levels()[byte_index] as u16) >> shift;
            if shift + bit_width > 8 {
                level |= (encoded.packed_levels()[byte_index + 1] as u16) << (8 - shift);
            }
            *rotated_value = self.codebook.level((level & mask) as u8);
            bit_offset += bit_width;
        }

        self.rotation
            .apply_transpose_into(&rotated_reconstruction, encoded.scale(), output);

        if let (Some(projector), Some(qjl)) = (&self.qjl, encoded.qjl()) {
            let mut residual = vec![0.0f32; self.config.dim()];
            projector.dequantize_into(qjl, &mut residual)?;
            for (value, residual_value) in output.iter_mut().zip(residual) {
                *value += residual_value;
            }
        }

        Ok(())
    }

    /// Plain score path used for correctness baselines.
    pub fn score_dot_plain(
        &self,
        query: &[f32],
        encoded: &TurboQuantVector,
    ) -> Result<f32, EncodingError> {
        if query.len() != self.config.dim() {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant query expected dim {}, got {}",
                self.config.dim(),
                query.len()
            )));
        }
        let reconstruction = self.dequantize(encoded)?;
        Ok(query
            .iter()
            .zip(&reconstruction)
            .map(|(&a, &b)| a * b)
            .sum())
    }

    /// SIMD-capable score path. The encoding is identical to the plain path;
    /// only the final dot product is accelerated.
    pub fn score_dot_simd(
        &self,
        query: &[f32],
        encoded: &TurboQuantVector,
    ) -> Result<f32, EncodingError> {
        if query.len() != self.config.dim() {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant query expected dim {}, got {}",
                self.config.dim(),
                query.len()
            )));
        }
        let reconstruction = self.dequantize(encoded)?;
        Ok(simd::dot(query, &reconstruction))
    }
}
