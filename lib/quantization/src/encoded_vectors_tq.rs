use std::alloc::Layout;
use std::borrow::Cow;
use std::mem::align_of;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use common::counter::hardware_counter::HardwareCounterCell;
use common::fs::atomic_save_json;
use common::mmap::MmapFlusher;
use common::typelevel::True;
use common::types::PointOffsetType;
use fs_err as fs;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::encoded_storage::{EncodedStorage, EncodedStorageBuilder};
use crate::encoded_vectors::{EncodedVectors, VectorParameters, validate_vector_parameters};
use crate::{DistanceType, EncodingError};

pub const DEFAULT_TURBO_QUANT_BITS: usize = 4;
const DEFAULT_TURBO_QUANT_SEED: u64 = 42;
const DEFAULT_HADAMARD_CHUNK_SIZE: usize = 128;
const DEFAULT_HADAMARD_ROUNDS: usize = 4;
const F32_SIZE: usize = std::mem::size_of::<f32>();
const QJL_CONST: f32 = 1.253_314_1;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum TqCorrection {
    NoCorrection,
    Qjl,
    #[default]
    Normalization,
    QjlNormalization,
}

impl TqCorrection {
    fn uses_qjl(self) -> bool {
        matches!(self, Self::Qjl | Self::QjlNormalization)
    }

    fn uses_norm_correction(self) -> bool {
        matches!(self, Self::Normalization | Self::QjlNormalization)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum TqRotation {
    NoRotation,
    #[default]
    Hadamard,
    Random,
}

pub struct EncodedVectorsTQ<TStorage: EncodedStorage> {
    encoded_vectors: TStorage,
    metadata: Metadata,
    metadata_path: Option<PathBuf>,
    codec: TurboQuantCodec,
}

/// Query-side cache for TurboQuant scoring.
///
/// The external query is still the original dense `f32` vector, but ADC-style
/// scoring needs a few derived representations to avoid recomputing them for
/// every scored point:
/// - `original_query`: fallback path in the original space
/// - `rotated_query`: main scalar-quantized ADC path
/// - `original_query_norm_sq`: L2 ADC path
/// - `qjl_projected_query`: QJL residual ADC path
pub struct EncodedQueryTQ {
    /// Original dense query in the source vector space.
    original_query: Vec<f32>,
    /// Query rotated into the same space as the scalar-quantized storage.
    rotated_query: Vec<f32>,
    /// Squared L2 norm of the original query, used by L2 ADC scoring.
    original_query_norm_sq: f32,
    /// Query projected into the QJL residual space.
    qjl_projected_query: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Metadata {
    pub vector_parameters: VectorParameters,
    pub bits: usize,
    pub correction: TqCorrection,
    pub rotation: TqRotation,
    pub hadamard_chunk: Option<usize>,
    #[serde(default = "default_turbo_quant_seed")]
    pub seed: u64,
}

fn default_turbo_quant_seed() -> u64 {
    DEFAULT_TURBO_QUANT_SEED
}

impl<TStorage: EncodedStorage> EncodedVectorsTQ<TStorage> {
    pub fn storage(&self) -> &TStorage {
        &self.encoded_vectors
    }

    /// Encode vector data
    ///
    /// # Arguments
    /// * `data` - iterator over original vector data
    /// * `storage_builder` - encoding result storage builder
    /// * `vector_parameters` - parameters of original vector data (dimension, distance, etc)
    /// * `count` - number of vectors in `data` iterator, used for progress bar
    /// * `bits` - number of bits for quantization (default: 4, range: 1-8)
    /// * `correction` - correction method
    /// * `rotation` - rotation method
    /// * `hadamard_chunk` - chunk size for structured Hadamard mixing (must be a power of two, default: 128)
    /// * `meta_path` - optional path to save metadata, if `None`, metadata will not be saved
    /// * `stopped` - Atomic bool that indicates if encoding should be stopped
    #[allow(clippy::too_many_arguments)]
    pub fn encode<'a>(
        data: impl Iterator<Item = impl AsRef<[f32]> + 'a> + Clone,
        mut storage_builder: impl EncodedStorageBuilder<Storage = TStorage>,
        vector_parameters: &VectorParameters,
        _count: usize,
        bits: usize,
        correction: TqCorrection,
        rotation: TqRotation,
        hadamard_chunk: Option<usize>,
        meta_path: Option<&Path>,
        stopped: &AtomicBool,
    ) -> Result<Self, EncodingError> {
        debug_assert!(validate_vector_parameters(data.clone(), vector_parameters).is_ok());

        let metadata = Metadata {
            vector_parameters: vector_parameters.clone(),
            bits,
            correction,
            rotation,
            hadamard_chunk,
            seed: DEFAULT_TURBO_QUANT_SEED,
        };
        let codec = TurboQuantCodec::new(&metadata)?;

        for vector in data {
            if stopped.load(Ordering::Relaxed) {
                return Err(EncodingError::Stopped);
            }

            let encoded_vector = codec.quantize(vector.as_ref())?;
            storage_builder
                .push_vector_data(&encoded_vector)
                .map_err(|e| {
                    EncodingError::EncodingError(format!("Failed to push encoded vector: {e}",))
                })?;
        }

        let encoded_vectors = storage_builder
            .build()
            .map_err(|e| EncodingError::EncodingError(format!("Failed to build storage: {e}",)))?;

        if let Some(meta_path) = meta_path {
            meta_path
                .parent()
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Path must have a parent directory",
                    )
                })
                .and_then(fs::create_dir_all)
                .map_err(|e| {
                    EncodingError::EncodingError(format!(
                        "Failed to create metadata directory: {e}",
                    ))
                })?;
            atomic_save_json(meta_path, &metadata).map_err(|e| {
                EncodingError::EncodingError(format!("Failed to save metadata: {e}",))
            })?;
        }

        Ok(Self {
            encoded_vectors,
            metadata,
            metadata_path: meta_path.map(PathBuf::from),
            codec,
        })
    }

    pub fn load(encoded_vectors: TStorage, meta_path: &Path) -> std::io::Result<Self> {
        let contents = fs::read_to_string(meta_path)?;
        let metadata: Metadata = serde_json::from_str(&contents)?;
        let codec = TurboQuantCodec::new(&metadata)
            .map_err(|err| std::io::Error::other(err.to_string()))?;
        Ok(Self {
            encoded_vectors,
            metadata,
            metadata_path: Some(meta_path.to_path_buf()),
            codec,
        })
    }

    // Get quantized vector size in bytes. We keep a fixed layout that always
    // reserves space for the optional QJL residual so storage sizing does not
    // depend on the correction mode.
    pub fn get_quantized_vector_size(vector_parameters: &VectorParameters, bits: usize) -> usize {
        let packed_levels = pack_len_bytes(vector_parameters.dim, bits as u8);
        packed_levels + F32_SIZE + qjl_signs_len_bytes(vector_parameters.dim) + F32_SIZE
    }

    fn score_point_simple(&self, query: &EncodedQueryTQ, vector: &[u8]) -> f32 {
        match self.codec.score_query(query, vector) {
            Ok(score) => {
                if self.metadata.vector_parameters.invert {
                    -score
                } else {
                    score
                }
            }
            Err(err) => panic!("Failed to score TurboQuant vector during scoring: {err}"),
        }
    }

    pub fn get_quantized_vector(&self, i: PointOffsetType) -> Cow<'_, [u8]> {
        self.encoded_vectors.get_vector_data(i)
    }

    pub fn layout(&self) -> Layout {
        Layout::from_size_align(self.quantized_vector_size(), align_of::<f32>()).unwrap()
    }

    pub fn get_metadata(&self) -> &Metadata {
        &self.metadata
    }
}

impl<TStorage: EncodedStorage> EncodedVectors for EncodedVectorsTQ<TStorage> {
    type EncodedQuery = EncodedQueryTQ;

    fn is_on_disk(&self) -> bool {
        self.encoded_vectors.is_on_disk()
    }

    fn encode_query(&self, query: &[f32]) -> EncodedQueryTQ {
        let rotated_query = self.codec.rotation.apply(query);
        let original_query_norm_sq = query.iter().map(|value| value * value).sum();
        let qjl_projected_query = self.codec.qjl.project(query);
        EncodedQueryTQ {
            original_query: query.to_vec(),
            rotated_query,
            original_query_norm_sq,
            qjl_projected_query,
        }
    }

    fn score_point(
        &self,
        query: &EncodedQueryTQ,
        i: PointOffsetType,
        hw_counter: &HardwareCounterCell,
    ) -> f32 {
        let encoded = self.encoded_vectors.get_vector_data(i);
        self.score_bytes(True, query, &encoded, hw_counter)
    }

    fn score_internal(
        &self,
        i: PointOffsetType,
        j: PointOffsetType,
        hw_counter: &HardwareCounterCell,
    ) -> f32 {
        let v1 = self.encoded_vectors.get_vector_data(i);
        let v2 = self.encoded_vectors.get_vector_data(j);
        hw_counter.vector_io_read().incr_delta(v1.len() + v2.len());

        let decoded1 = self
            .codec
            .dequantize(&v1)
            .unwrap_or_else(|err| panic!("Failed to decode TurboQuant vector {i}: {err}"));
        let decoded2 = self
            .codec
            .dequantize(&v2)
            .unwrap_or_else(|err| panic!("Failed to decode TurboQuant vector {j}: {err}"));

        score_vectors(
            &decoded1,
            &decoded2,
            self.metadata.vector_parameters.distance_type,
            self.metadata.vector_parameters.invert,
        )
    }

    fn quantized_vector_size(&self) -> usize {
        Self::get_quantized_vector_size(&self.metadata.vector_parameters, self.metadata.bits)
    }

    fn encode_internal_vector(&self, id: PointOffsetType) -> Option<EncodedQueryTQ> {
        let encoded = self.encoded_vectors.get_vector_data(id);
        let query = self.codec.dequantize(&encoded).ok()?;
        Some(self.encode_query(&query))
    }

    fn upsert_vector(
        &mut self,
        id: PointOffsetType,
        vector: &[f32],
        hw_counter: &HardwareCounterCell,
    ) -> std::io::Result<()> {
        let encoded = self
            .codec
            .quantize(vector)
            .map_err(|err| std::io::Error::other(err.to_string()))?;
        self.encoded_vectors.upsert_vector(id, &encoded, hw_counter)
    }

    fn vectors_count(&self) -> usize {
        self.encoded_vectors.vectors_count()
    }

    fn flusher(&self) -> MmapFlusher {
        self.encoded_vectors.flusher()
    }

    fn files(&self) -> Vec<PathBuf> {
        let mut files = self.encoded_vectors.files();
        if let Some(meta_path) = &self.metadata_path {
            files.push(meta_path.clone());
        }
        files
    }

    fn immutable_files(&self) -> Vec<PathBuf> {
        let mut files = self.encoded_vectors.immutable_files();
        if let Some(meta_path) = &self.metadata_path {
            files.push(meta_path.clone());
        }
        files
    }

    type SupportsBytes = True;

    fn score_bytes(
        &self,
        _: Self::SupportsBytes,
        query: &Self::EncodedQuery,
        bytes: &[u8],
        hw_counter: &HardwareCounterCell,
    ) -> f32 {
        hw_counter.cpu_counter().incr_delta(bytes.len());
        self.score_point_simple(query, bytes)
    }
}

fn score_vectors(lhs: &[f32], rhs: &[f32], distance_type: DistanceType, invert: bool) -> f32 {
    let mut result = 0.0f32;
    for (&a, &b) in lhs.iter().zip(rhs) {
        match distance_type {
            DistanceType::Dot => {
                result += a * b;
            }
            DistanceType::L1 => {
                result += (a - b).abs();
            }
            DistanceType::L2 => {
                let diff = a - b;
                result += diff * diff;
            }
        }
    }

    if invert { -result } else { result }
}

struct TurboQuantCodec {
    dim: usize,
    bits: u8,
    distance_type: DistanceType,
    correction: TqCorrection,
    rotation: Rotation,
    codebook: QuantizationCodebook,
    qjl: QjlProjector,
}

impl TurboQuantCodec {
    fn new(metadata: &Metadata) -> Result<Self, EncodingError> {
        let dim = metadata.vector_parameters.dim;
        if dim == 0 {
            return Err(EncodingError::ArgumentsError(
                "TurboQuant dimension must be non-zero".to_owned(),
            ));
        }
        if metadata.rotation == TqRotation::Hadamard {
            let chunk_size = metadata
                .hadamard_chunk
                .unwrap_or(DEFAULT_HADAMARD_CHUNK_SIZE);
            if chunk_size == 0 || !chunk_size.is_power_of_two() {
                return Err(EncodingError::ArgumentsError(format!(
                    "TurboQuant Hadamard chunk size must be a non-zero power of two, got {chunk_size}"
                )));
            }
        }

        let bits = u8::try_from(metadata.bits).map_err(|_| {
            EncodingError::ArgumentsError(format!(
                "TurboQuant bit width must fit into u8, got {}",
                metadata.bits
            ))
        })?;
        if !(1..=8).contains(&bits) {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant bit width must be in 1..=8, got {}",
                metadata.bits
            )));
        }

        Ok(Self {
            dim,
            bits,
            distance_type: metadata.vector_parameters.distance_type,
            correction: metadata.correction,
            rotation: Rotation::new(
                metadata.rotation,
                dim,
                metadata.seed,
                metadata
                    .hadamard_chunk
                    .unwrap_or(DEFAULT_HADAMARD_CHUNK_SIZE),
            ),
            codebook: QuantizationCodebook::new(bits, dim),
            qjl: QjlProjector::new(dim, metadata.seed ^ 0x5bf0_3635_d4f9_8a51),
        })
    }

    fn packed_levels_len(&self) -> usize {
        pack_len_bytes(self.dim, self.bits)
    }

    fn encoded_len(&self) -> usize {
        self.packed_levels_len() + F32_SIZE + qjl_signs_len_bytes(self.dim) + F32_SIZE
    }

    fn quantize(&self, vector: &[f32]) -> Result<Vec<u8>, EncodingError> {
        if vector.len() != self.dim {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant expected dim {}, got {}",
                self.dim,
                vector.len()
            )));
        }

        let original_norm = l2_norm(vector);
        let safe_norm = if original_norm > 0.0 {
            original_norm
        } else {
            1.0
        };
        let normalized: Vec<f32> = vector.iter().map(|&value| value / safe_norm).collect();
        let rotated = self.rotation.apply(&normalized);

        let mut levels = Vec::with_capacity(self.dim);
        let mut rotated_reconstruction = Vec::with_capacity(self.dim);
        for &value in &rotated {
            let level = self.codebook.nearest_index(value);
            levels.push(level);
            rotated_reconstruction.push(self.codebook.level(level));
        }

        let reconstructed_rotated_norm = l2_norm(&rotated_reconstruction);
        let scale = if self.correction.uses_norm_correction() {
            let safe_recon_norm = if reconstructed_rotated_norm > 1e-12 {
                reconstructed_rotated_norm
            } else {
                1.0
            };
            original_norm / safe_recon_norm
        } else {
            original_norm
        };

        let packed_levels = pack_bits(&levels, self.bits);
        let mut encoded = Vec::with_capacity(self.encoded_len());
        encoded.extend_from_slice(&packed_levels);
        encoded.extend_from_slice(&scale.to_ne_bytes());

        if self.correction.uses_qjl() {
            let base_reconstruction = self
                .rotation
                .apply_transpose(&rotated_reconstruction, scale);
            let residual: Vec<f32> = vector
                .iter()
                .zip(&base_reconstruction)
                .map(|(&value, &hat)| value - hat)
                .collect();
            let (packed_signs, residual_norm) = self.qjl.quantize(&residual);
            encoded.extend_from_slice(&packed_signs);
            encoded.extend_from_slice(&residual_norm.to_ne_bytes());
        } else {
            encoded.resize(encoded.len() + qjl_signs_len_bytes(self.dim), 0);
            encoded.extend_from_slice(&0.0f32.to_ne_bytes());
        }

        debug_assert_eq!(encoded.len(), self.encoded_len());
        Ok(encoded)
    }

    fn dequantize(&self, encoded: &[u8]) -> Result<Vec<f32>, EncodingError> {
        if encoded.len() != self.encoded_len() {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant encoded vector expected {} bytes, got {}",
                self.encoded_len(),
                encoded.len()
            )));
        }

        let packed_levels_len = self.packed_levels_len();
        let packed_levels = &encoded[..packed_levels_len];
        let scale_start = packed_levels_len;
        let scale_end = scale_start + F32_SIZE;
        let scale = f32::from_ne_bytes(
            encoded[scale_start..scale_end]
                .try_into()
                .expect("scale is stored as four bytes"),
        );
        if scale.is_sign_negative() {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant scale must be non-negative, got {scale}"
            )));
        }

        let levels = unpack_bits(packed_levels, self.bits, self.dim);
        let rotated_reconstruction: Vec<f32> = levels
            .iter()
            .map(|&level| self.codebook.level(level))
            .collect();
        let mut base = self
            .rotation
            .apply_transpose(&rotated_reconstruction, scale);

        if self.correction.uses_qjl() {
            let packed_signs_start = scale_end;
            let packed_signs_end = packed_signs_start + qjl_signs_len_bytes(self.dim);
            let packed_signs = &encoded[packed_signs_start..packed_signs_end];
            let residual_norm = f32::from_ne_bytes(
                encoded[packed_signs_end..packed_signs_end + F32_SIZE]
                    .try_into()
                    .expect("residual norm is stored as four bytes"),
            );
            let residual = self.qjl.dequantize(packed_signs, residual_norm)?;
            for (value, residual_value) in base.iter_mut().zip(residual) {
                *value += residual_value;
            }
        }

        Ok(base)
    }

    fn score_query(&self, query: &EncodedQueryTQ, encoded: &[u8]) -> Result<f32, EncodingError> {
        let parsed = self.parse_encoded_vector(encoded)?;
        match self.distance_type {
            DistanceType::Dot => self.score_dot_adc(query, &parsed),
            DistanceType::L2 if !self.correction.uses_qjl() => self.score_l2_adc(query, &parsed),
            DistanceType::L1 | DistanceType::L2 => {
                let decoded = self.dequantize(encoded)?;
                Ok(score_vectors(
                    &query.original_query,
                    &decoded,
                    self.distance_type,
                    false,
                ))
            }
        }
    }

    fn score_dot_adc(
        &self,
        query: &EncodedQueryTQ,
        parsed: &ParsedEncodedVector<'_>,
    ) -> Result<f32, EncodingError> {
        let mut dot_sum = 0.0f32;
        self.visit_levels(parsed.packed_levels, |dim, level_idx| {
            dot_sum += query.rotated_query[dim] * self.codebook.level(level_idx);
        })?;

        let mut score = parsed.scale * dot_sum;
        if self.correction.uses_qjl() {
            let residual_dot = self.qjl.dot_projected_signs(
                &query.qjl_projected_query,
                parsed.qjl_signs,
                parsed.residual_norm,
            )?;
            score += residual_dot;
        }
        Ok(score)
    }

    fn score_l2_adc(
        &self,
        query: &EncodedQueryTQ,
        parsed: &ParsedEncodedVector<'_>,
    ) -> Result<f32, EncodingError> {
        let mut dot_sum = 0.0f32;
        let mut level_norm2 = 0.0f32;
        self.visit_levels(parsed.packed_levels, |dim, level_idx| {
            let level = self.codebook.level(level_idx);
            dot_sum += query.rotated_query[dim] * level;
            level_norm2 += level * level;
        })?;

        Ok(
            query.original_query_norm_sq + parsed.scale * parsed.scale * level_norm2
                - 2.0 * parsed.scale * dot_sum,
        )
    }

    fn visit_levels(
        &self,
        packed_levels: &[u8],
        mut visitor: impl FnMut(usize, u8),
    ) -> Result<(), EncodingError> {
        let mask = if self.bits == 8 {
            u16::MAX
        } else {
            (1u16 << self.bits) - 1
        };
        let mut bit_offset = 0usize;
        for dim in 0..self.dim {
            let byte_index = bit_offset / 8;
            let shift = bit_offset % 8;
            let mut value = (packed_levels.get(byte_index).copied().ok_or_else(|| {
                EncodingError::ArgumentsError("TurboQuant packed levels are truncated".to_owned())
            })? as u16)
                >> shift;
            if shift + self.bits as usize > 8 {
                value |= (packed_levels.get(byte_index + 1).copied().ok_or_else(|| {
                    EncodingError::ArgumentsError(
                        "TurboQuant packed levels cross a missing byte boundary".to_owned(),
                    )
                })? as u16)
                    << (8 - shift);
            }
            visitor(dim, (value & mask) as u8);
            bit_offset += self.bits as usize;
        }
        Ok(())
    }

    fn parse_encoded_vector<'a>(
        &self,
        encoded: &'a [u8],
    ) -> Result<ParsedEncodedVector<'a>, EncodingError> {
        if encoded.len() != self.encoded_len() {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant encoded vector expected {} bytes, got {}",
                self.encoded_len(),
                encoded.len()
            )));
        }

        let packed_levels_len = self.packed_levels_len();
        let packed_levels = &encoded[..packed_levels_len];
        let scale_start = packed_levels_len;
        let scale_end = scale_start + F32_SIZE;
        let scale = f32::from_ne_bytes(
            encoded[scale_start..scale_end]
                .try_into()
                .expect("scale is stored as four bytes"),
        );
        if scale.is_sign_negative() {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant scale must be non-negative, got {scale}"
            )));
        }

        let qjl_signs_start = scale_end;
        let qjl_signs_end = qjl_signs_start + qjl_signs_len_bytes(self.dim);
        let qjl_signs = &encoded[qjl_signs_start..qjl_signs_end];
        let residual_norm = f32::from_ne_bytes(
            encoded[qjl_signs_end..qjl_signs_end + F32_SIZE]
                .try_into()
                .expect("residual norm is stored as four bytes"),
        );

        Ok(ParsedEncodedVector {
            packed_levels,
            scale,
            qjl_signs,
            residual_norm,
        })
    }
}

struct ParsedEncodedVector<'a> {
    packed_levels: &'a [u8],
    scale: f32,
    qjl_signs: &'a [u8],
    residual_norm: f32,
}

#[derive(Clone)]
enum Rotation {
    Identity,
    Haar(HaarRotation),
    Hadamard(HadamardRotation),
}

impl Rotation {
    fn new(kind: TqRotation, dim: usize, seed: u64, hadamard_chunk_size: usize) -> Self {
        match kind {
            TqRotation::NoRotation => Self::Identity,
            TqRotation::Random => Self::Haar(HaarRotation::new(dim, seed)),
            TqRotation::Hadamard => Self::Hadamard(HadamardRotation::new(
                dim,
                seed,
                hadamard_chunk_size,
                DEFAULT_HADAMARD_ROUNDS,
            )),
        }
    }

    fn apply(&self, input: &[f32]) -> Vec<f32> {
        match self {
            Self::Identity => input.to_vec(),
            Self::Haar(rotation) => rotation.apply(input),
            Self::Hadamard(rotation) => rotation.apply(input),
        }
    }

    fn apply_transpose(&self, input: &[f32], scale: f32) -> Vec<f32> {
        match self {
            Self::Identity => input.iter().map(|&value| value * scale).collect(),
            Self::Haar(rotation) => rotation.apply_transpose(input, scale),
            Self::Hadamard(rotation) => rotation.apply_transpose(input, scale),
        }
    }
}

#[derive(Clone)]
struct HaarRotation {
    dim: usize,
    matrix: Vec<f32>,
}

impl HaarRotation {
    fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut matrix = vec![0.0f32; dim * dim];
        let mut filled = 0usize;
        while filled < matrix.len() {
            let (z0, z1) = sample_standard_normal_pair(&mut rng);
            matrix[filled] = z0;
            filled += 1;
            if filled < matrix.len() {
                matrix[filled] = z1;
                filled += 1;
            }
        }

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
struct HadamardRotation {
    dim: usize,
    chunk_size: usize,
    chunk_scale: f32,
    rounds: Vec<HadamardRound>,
}

#[derive(Clone)]
struct HadamardRound {
    left_signs: Vec<f32>,
    right_signs: Vec<f32>,
    permutation: Vec<usize>,
    inverse_permutation: Vec<usize>,
}

impl HadamardRotation {
    fn new(dim: usize, seed: u64, chunk_size: usize, rounds: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let rounds = (0..rounds)
            .map(|_| HadamardRound::new(dim, &mut rng))
            .collect();
        Self {
            dim,
            chunk_size,
            chunk_scale: 1.0 / (chunk_size as f32).sqrt(),
            rounds,
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

    fn apply(&self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        let mut next = vec![0.0f32; self.dim];
        let mut tmp = vec![0.0f32; self.dim];
        let mut scratch = vec![0.0f32; self.chunk_size];
        for round in &self.rounds {
            self.apply_round(round, &current, &mut next, &mut tmp, &mut scratch);
            std::mem::swap(&mut current, &mut next);
        }
        current
    }

    fn apply_transpose(&self, input: &[f32], scale: f32) -> Vec<f32> {
        let mut current = input.to_vec();
        let mut next = vec![0.0f32; self.dim];
        let mut tmp = vec![0.0f32; self.dim];
        let mut scratch = vec![0.0f32; self.chunk_size];
        for round in self.rounds.iter().rev() {
            self.apply_round_transpose(round, &current, &mut next, &mut tmp, &mut scratch);
            std::mem::swap(&mut current, &mut next);
        }
        current.iter_mut().for_each(|value| *value *= scale);
        current
    }

    fn apply_round(
        &self,
        round: &HadamardRound,
        input: &[f32],
        output: &mut [f32],
        tmp: &mut [f32],
        scratch: &mut [f32],
    ) {
        for chunk_start in (0..self.dim).step_by(self.chunk_size) {
            let chunk_len = usize::min(self.chunk_size, self.dim - chunk_start);
            scratch.fill(0.0);
            for offset in 0..chunk_len {
                scratch[offset] =
                    input[chunk_start + offset] * round.right_signs[chunk_start + offset];
            }
            Self::fwht(scratch);
            for offset in 0..chunk_len {
                tmp[chunk_start + offset] =
                    scratch[offset] * round.left_signs[chunk_start + offset] * self.chunk_scale;
            }
        }

        for (src, &dst) in round.permutation.iter().enumerate() {
            output[dst] = tmp[src];
        }
    }

    fn apply_round_transpose(
        &self,
        round: &HadamardRound,
        input: &[f32],
        output: &mut [f32],
        tmp: &mut [f32],
        scratch: &mut [f32],
    ) {
        for (dst, &src) in round.inverse_permutation.iter().enumerate() {
            tmp[src] = input[dst];
        }

        for chunk_start in (0..self.dim).step_by(self.chunk_size) {
            let chunk_len = usize::min(self.chunk_size, self.dim - chunk_start);
            scratch.fill(0.0);
            for offset in 0..chunk_len {
                scratch[offset] =
                    tmp[chunk_start + offset] * round.left_signs[chunk_start + offset];
            }
            Self::fwht(scratch);
            for offset in 0..chunk_len {
                output[chunk_start + offset] =
                    scratch[offset] * round.right_signs[chunk_start + offset] * self.chunk_scale;
            }
        }
    }
}

impl HadamardRound {
    fn new(dim: usize, rng: &mut StdRng) -> Self {
        let left_signs = (0..dim)
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let right_signs = (0..dim)
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let mut permutation = (0..dim).collect::<Vec<_>>();
        permutation.shuffle(rng);
        let mut inverse_permutation = vec![0usize; dim];
        for (src, &dst) in permutation.iter().enumerate() {
            inverse_permutation[dst] = src;
        }
        Self {
            left_signs,
            right_signs,
            permutation,
            inverse_permutation,
        }
    }
}

#[derive(Clone)]
struct QuantizationCodebook {
    levels: Vec<f32>,
    boundaries: Vec<f32>,
}

impl QuantizationCodebook {
    fn new(bit_width: u8, dim: usize) -> Self {
        let levels = Self::optimal_centroids(bit_width, dim);
        let boundaries = levels
            .windows(2)
            .map(|pair| 0.5 * (pair[0] + pair[1]))
            .collect();
        Self { levels, boundaries }
    }

    fn level(&self, index: u8) -> f32 {
        self.levels[index as usize]
    }

    fn nearest_index(&self, value: f32) -> u8 {
        self.boundaries
            .partition_point(|boundary| *boundary < value) as u8
    }

    fn optimal_centroids(bit_width: u8, dim: usize) -> Vec<f32> {
        let sigma = 1.0 / (dim as f64).sqrt();
        match bit_width {
            1 => {
                let c = (2.0 / (std::f64::consts::PI * dim as f64)).sqrt() as f32;
                vec![-c, c]
            }
            2 => vec![-1.51, -0.453, 0.453, 1.51]
                .into_iter()
                .map(|value| (value / (dim as f64).sqrt()) as f32)
                .collect(),
            _ => Self::lloyds_gaussian(1usize << bit_width, sigma),
        }
    }

    fn lloyds_gaussian(level_count: usize, sigma: f64) -> Vec<f32> {
        let mut boundaries: Vec<f64> = (1..level_count)
            .map(|index| inverse_normal_cdf(index as f64 / level_count as f64) * sigma)
            .collect();

        let mut centroids = vec![0.0f64; level_count];
        Self::update_centroids(&mut centroids, &boundaries, sigma);

        for _ in 0..100 {
            for (boundary, pair) in boundaries.iter_mut().zip(centroids.windows(2)) {
                *boundary = 0.5 * (pair[0] + pair[1]);
            }
            Self::update_centroids(&mut centroids, &boundaries, sigma);
        }

        centroids.into_iter().map(|value| value as f32).collect()
    }

    fn update_centroids(centroids: &mut [f64], boundaries: &[f64], sigma: f64) {
        if centroids.is_empty() {
            return;
        }

        centroids[0] = gaussian_conditional_expectation(sigma, f64::NEG_INFINITY, boundaries[0]);
        for index in 1..centroids.len().saturating_sub(1) {
            centroids[index] =
                gaussian_conditional_expectation(sigma, boundaries[index - 1], boundaries[index]);
        }
        centroids[centroids.len() - 1] = gaussian_conditional_expectation(
            sigma,
            boundaries[boundaries.len() - 1],
            f64::INFINITY,
        );
    }
}

#[derive(Clone)]
struct QjlProjector {
    dim: usize,
    matrix: Vec<f32>,
}

impl QjlProjector {
    fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut matrix = Vec::with_capacity(dim * dim);
        while matrix.len() < dim * dim {
            let (z0, z1) = sample_standard_normal_pair(&mut rng);
            matrix.push(z0);
            if matrix.len() < dim * dim {
                matrix.push(z1);
            }
        }
        Self { dim, matrix }
    }

    fn quantize(&self, residual: &[f32]) -> (Vec<u8>, f32) {
        let norm = l2_norm(residual);
        let projections = self.project(residual);
        let signs: Vec<u8> = projections
            .into_iter()
            .map(|projection| u8::from(projection >= 0.0))
            .collect();

        (pack_bits(&signs, 1), norm)
    }

    fn project(&self, vector: &[f32]) -> Vec<f32> {
        self.matrix
            .chunks_exact(self.dim)
            .map(|row| row.iter().zip(vector).map(|(&a, &b)| a * b).sum::<f32>())
            .collect()
    }

    fn dequantize(&self, packed_signs: &[u8], norm: f32) -> Result<Vec<f32>, EncodingError> {
        let expected_bytes = qjl_signs_len_bytes(self.dim);
        if packed_signs.len() != expected_bytes {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant QJL residual expected {expected_bytes} bytes, got {}",
                packed_signs.len()
            )));
        }
        if norm.is_sign_negative() {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant QJL residual norm must be non-negative, got {norm}"
            )));
        }
        if norm == 0.0 {
            return Ok(vec![0.0; self.dim]);
        }

        let signs = unpack_bits(packed_signs, 1, self.dim);
        let mut reconstruction = vec![0.0f32; self.dim];
        for row in 0..self.dim {
            let sign = if signs[row] == 0 { -1.0 } else { 1.0 };
            let row_slice = &self.matrix[row * self.dim..(row + 1) * self.dim];
            for (out, &value) in reconstruction.iter_mut().zip(row_slice) {
                *out += sign * value;
            }
        }

        let scale = QJL_CONST * norm / self.dim as f32;
        reconstruction.iter_mut().for_each(|value| *value *= scale);
        Ok(reconstruction)
    }

    fn dot_projected_signs(
        &self,
        projected_query: &[f32],
        packed_signs: &[u8],
        norm: f32,
    ) -> Result<f32, EncodingError> {
        let expected_bytes = qjl_signs_len_bytes(self.dim);
        if packed_signs.len() != expected_bytes {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant QJL residual expected {expected_bytes} bytes, got {}",
                packed_signs.len()
            )));
        }
        if projected_query.len() != self.dim {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant projected query expected dim {}, got {}",
                self.dim,
                projected_query.len()
            )));
        }
        if norm.is_sign_negative() {
            return Err(EncodingError::ArgumentsError(format!(
                "TurboQuant QJL residual norm must be non-negative, got {norm}"
            )));
        }
        if norm == 0.0 {
            return Ok(0.0);
        }

        let signs = unpack_bits(packed_signs, 1, self.dim);
        let signed_projection_sum = signs
            .into_iter()
            .zip(projected_query)
            .map(|(sign, projection)| if sign == 0 { -projection } else { *projection })
            .sum::<f32>();
        Ok(QJL_CONST * norm / self.dim as f32 * signed_projection_sum)
    }
}

fn pack_len_bytes(value_count: usize, bit_width: u8) -> usize {
    (value_count * bit_width as usize).div_ceil(8)
}

fn qjl_signs_len_bytes(dim: usize) -> usize {
    dim.div_ceil(8)
}

fn pack_bits(values: &[u8], bit_width: u8) -> Vec<u8> {
    debug_assert!((1..=8).contains(&bit_width));

    let total_bits = values.len() * bit_width as usize;
    let mut packed = vec![0u8; total_bits.div_ceil(8)];
    let mask = if bit_width == 8 {
        u16::MAX
    } else {
        (1u16 << bit_width) - 1
    };

    let mut bit_offset = 0usize;
    for &value in values {
        let value = (value as u16) & mask;
        let byte_index = bit_offset / 8;
        let shift = bit_offset % 8;

        packed[byte_index] |= (value << shift) as u8;
        if shift + bit_width as usize > 8 {
            packed[byte_index + 1] |= (value >> (8 - shift)) as u8;
        }

        bit_offset += bit_width as usize;
    }

    packed
}

fn unpack_bits(packed: &[u8], bit_width: u8, value_count: usize) -> Vec<u8> {
    debug_assert!((1..=8).contains(&bit_width));

    let mask = if bit_width == 8 {
        u16::MAX
    } else {
        (1u16 << bit_width) - 1
    };

    let mut output = Vec::with_capacity(value_count);
    let mut bit_offset = 0usize;
    for _ in 0..value_count {
        let byte_index = bit_offset / 8;
        let shift = bit_offset % 8;
        let mut value = (packed[byte_index] as u16) >> shift;
        if shift + bit_width as usize > 8 {
            value |= (packed[byte_index + 1] as u16) << (8 - shift);
        }
        output.push((value & mask) as u8);
        bit_offset += bit_width as usize;
    }
    output
}

fn l2_norm(values: &[f32]) -> f32 {
    values
        .iter()
        .map(|&value| value * value)
        .sum::<f32>()
        .sqrt()
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(&a, &b)| a * b).sum()
}

fn sample_standard_normal_pair(rng: &mut impl RngExt) -> (f32, f32) {
    let u1 = (1.0f32 - rng.random::<f32>()).max(1e-12f32);
    let u2 = rng.random::<f32>();
    let radius = (-2.0f32 * u1.ln()).sqrt();
    let theta = 2.0f32 * std::f32::consts::PI * u2;
    (radius * theta.cos(), radius * theta.sin())
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

fn inverse_normal_cdf(p: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&p));
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let plow = 0.02425;
    let phigh = 1.0 - plow;

    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p > phigh {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

fn gaussian_conditional_expectation(sigma: f64, a: f64, b: f64) -> f64 {
    let a_std = if a.is_finite() { a / sigma } else { a };
    let b_std = if b.is_finite() { b / sigma } else { b };

    let probability = if !a_std.is_finite() {
        normal_cdf(b_std)
    } else if !b_std.is_finite() {
        1.0 - normal_cdf(a_std)
    } else {
        normal_cdf(b_std) - normal_cdf(a_std)
    };

    if probability < 1e-15 {
        return if a.is_finite() && !b.is_finite() {
            a + sigma
        } else if !a.is_finite() && b.is_finite() {
            b - sigma
        } else if a.is_finite() && b.is_finite() {
            0.5 * (a + b)
        } else {
            0.0
        };
    }

    sigma * (normal_pdf(a_std) - normal_pdf(b_std)) / probability
}
