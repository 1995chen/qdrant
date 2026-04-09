//! Recall helpers used by the standalone TurboQuant tests.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use super::{TurboQuantCodec, TurboQuantVector};
use crate::EncodingError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallAtK {
    pub k: usize,
    pub recall: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecallReport {
    pub entries: Vec<RecallAtK>,
}

impl RecallReport {
    pub fn recall(&self, k: usize) -> Option<f32> {
        self.entries
            .iter()
            .find(|entry| entry.k == k)
            .map(|entry| entry.recall)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecallEvaluation {
    pub report: RecallReport,
    pub elapsed: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactSearchBaseline {
    pub report: RecallReport,
    pub top_k_per_query: Vec<Vec<usize>>,
    pub elapsed: Duration,
}

impl ExactSearchBaseline {
    pub fn max_k(&self) -> usize {
        self.top_k_per_query.first().map_or(0, Vec::len)
    }
}

pub fn compute_exact_baseline(
    original: &[Vec<f32>],
    queries: &[Vec<f32>],
    ks: &[usize],
) -> ExactSearchBaseline {
    let max_k = ks.iter().copied().max().unwrap_or(0);
    let started = Instant::now();
    let mut top_k_per_query = Vec::with_capacity(queries.len());

    for query in queries {
        let mut exact_scores: Vec<(usize, f32)> = original
            .iter()
            .enumerate()
            .map(|(index, vector)| (index, dot(query, vector)))
            .collect();
        exact_scores.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
        top_k_per_query.push(
            exact_scores
                .iter()
                .take(max_k)
                .map(|&(index, _)| index)
                .collect(),
        );
    }

    let report = RecallReport {
        entries: ks.iter().map(|&k| RecallAtK { k, recall: 1.0 }).collect(),
    };

    ExactSearchBaseline {
        report,
        top_k_per_query,
        elapsed: started.elapsed(),
    }
}

pub fn evaluate_recall_with_baseline(
    codec: &TurboQuantCodec,
    encoded: &[TurboQuantVector],
    queries: &[Vec<f32>],
    ks: &[usize],
    use_simd: bool,
    baseline: &ExactSearchBaseline,
) -> Result<RecallEvaluation, EncodingError> {
    assert!(
        ks.iter().copied().max().unwrap_or(0) <= baseline.max_k(),
        "baseline must be computed with max_k >= requested ks"
    );

    let mut hit_counts: BTreeMap<usize, usize> = ks.iter().copied().map(|k| (k, 0)).collect();
    let total = queries.len();
    let started = Instant::now();

    for (query, exact_top_k) in queries.iter().zip(&baseline.top_k_per_query) {
        let mut approx_scores: Vec<(usize, f32)> = encoded
            .iter()
            .enumerate()
            .map(|(index, vector)| {
                let score = if use_simd {
                    codec.score_dot_simd(query, vector)
                } else {
                    codec.score_dot_plain(query, vector)
                }?;
                Ok((index, score))
            })
            .collect::<Result<Vec<_>, EncodingError>>()?;
        approx_scores.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));

        for &k in ks {
            let exact: BTreeSet<_> = exact_top_k.iter().take(k).copied().collect();
            let approx: BTreeSet<_> = approx_scores
                .iter()
                .take(k)
                .map(|&(index, _)| index)
                .collect();
            let hits = exact.intersection(&approx).count();
            *hit_counts.get_mut(&k).expect("k should exist") += hits;
        }
    }

    let report = RecallReport {
        entries: ks
            .iter()
            .map(|&k| RecallAtK {
                k,
                recall: hit_counts[&k] as f32 / (total * k) as f32,
            })
            .collect(),
    };

    Ok(RecallEvaluation {
        report,
        elapsed: started.elapsed(),
    })
}

pub fn evaluate_recall(
    codec: &TurboQuantCodec,
    encoded: &[TurboQuantVector],
    original: &[Vec<f32>],
    queries: &[Vec<f32>],
    ks: &[usize],
    use_simd: bool,
) -> Result<RecallReport, EncodingError> {
    let baseline = compute_exact_baseline(original, queries, ks);
    Ok(evaluate_recall_with_baseline(codec, encoded, queries, ks, use_simd, &baseline)?.report)
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(&a, &b)| a * b).sum()
}
