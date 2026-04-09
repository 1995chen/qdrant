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
mod codec;
mod config;
mod math;
mod packing;
mod qjl;
mod recall;
mod rotation;
mod simd;
mod vector;

pub use codec::TurboQuantCodec;
pub use config::{NormCorrection, TurboQuantConfig};
pub use qjl::QjlResidual;
pub use recall::{
    ExactSearchBaseline, RecallAtK, RecallEvaluation, RecallReport, compute_exact_baseline,
    evaluate_recall, evaluate_recall_with_baseline,
};
pub use rotation::RotationKind;
pub use vector::TurboQuantVector;
