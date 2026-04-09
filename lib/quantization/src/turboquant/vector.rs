use super::{QjlResidual, TurboQuantConfig};
use crate::EncodingError;

/// Encoded representation of one vector.
///
/// `packed_levels` always contains the level stage. `qjl` is only populated
/// when the codec was created with QJL enabled in the config.
#[derive(Clone, Debug, PartialEq)]
pub struct TurboQuantVector {
    packed_levels: Vec<u8>,
    scale: f32,
    qjl: Option<QjlResidual>,
}

impl TurboQuantVector {
    pub fn packed_len_bytes(dim: usize, bit_width: u8) -> usize {
        (dim * bit_width as usize).div_ceil(8)
    }

    pub(crate) fn new(packed_levels: Vec<u8>, scale: f32, qjl: Option<QjlResidual>) -> Self {
        Self {
            packed_levels,
            scale,
            qjl,
        }
    }

    pub fn packed_levels(&self) -> &[u8] {
        &self.packed_levels
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn qjl(&self) -> Option<&QjlResidual> {
        self.qjl.as_ref()
    }

    pub(crate) fn validate(&self, config: &TurboQuantConfig) -> Result<(), EncodingError> {
        let expected_level_bytes = Self::packed_len_bytes(config.dim(), config.bit_width());
        if self.packed_levels.len() != expected_level_bytes {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant encoded levels expected {expected_level_bytes} bytes, got {}",
                self.packed_levels.len()
            )));
        }
        if self.scale.is_sign_negative() {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant scale must be non-negative, got {}",
                self.scale
            )));
        }

        match (&self.qjl, config.qjl()) {
            (Some(residual), true) => residual.validate(config.dim())?,
            (None, false) => {}
            (Some(_), false) => {
                return Err(EncodingError::ArgumentsError(
                    "TurboQuant vector has QJL residual but codec config has QJL disabled"
                        .to_owned(),
                ));
            }
            (None, true) => {
                return Err(EncodingError::ArgumentsError(
                    "TurboQuant vector is missing QJL residual for a codec with QJL enabled"
                        .to_owned(),
                ));
            }
        }

        Ok(())
    }
}
