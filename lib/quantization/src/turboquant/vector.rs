use super::QjlResidual;

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
}
