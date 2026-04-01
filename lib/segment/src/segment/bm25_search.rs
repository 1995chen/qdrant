use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

use common::counter::hardware_counter::HardwareCounterCell;
use common::top_k::TopK;
use common::types::{PointOffsetType, ScoredPointOffset};

use super::Segment;
use crate::common::check_stopped;
use crate::common::operation_error::OperationResult;
use crate::json_path::JsonPath;
use crate::payload_storage::FilterContext;
use crate::types::{Filter, PayloadContainer};

impl Segment {
    pub(super) fn do_search_bm25(
        &self,
        field: &JsonPath,
        query: &str,
        filter: Option<&Filter>,
        top: usize,
        score_threshold: Option<f32>,
        is_stopped: &AtomicBool,
        hw_counter: &HardwareCounterCell,
        k1: f32,
        b: f32,
    ) -> OperationResult<Vec<ScoredPointOffset>> {
        if top == 0 {
            return Ok(Vec::new());
        }

        let payload_index = self.payload_index.borrow();
        let text_index = payload_index.get_full_text_index(field)?;
        let Some(query_tokens) = text_index.parse_bm25_query(query, hw_counter) else {
            return Ok(Vec::new());
        };

        let avgdl = text_index.average_len();
        if avgdl <= 0.0 {
            return Ok(Vec::new());
        }

        let filter_context = filter
            .map(|filter| payload_index.struct_filtered_context(filter, hw_counter))
            .transpose()?;

        let mut top_k = TopK::new(top);

        for point_id in text_index.bm25_candidates(query, hw_counter) {
            check_stopped(is_stopped)?;

            if filter_context
                .as_ref()
                .is_some_and(|context| !context.check(point_id))
            {
                continue;
            }

            let score = self.score_bm25_point(
                field,
                point_id,
                text_index,
                &query_tokens,
                avgdl,
                hw_counter,
                k1,
                b,
            )?;

            if score > 0.0 && score_threshold.is_none_or(|threshold| score >= threshold) {
                top_k.push(ScoredPointOffset {
                    idx: point_id,
                    score,
                });
            }
        }

        Ok(top_k.into_vec())
    }

    #[allow(clippy::too_many_arguments)]
    fn score_bm25_point(
        &self,
        field: &JsonPath,
        point_id: PointOffsetType,
        text_index: &crate::index::field_index::full_text_index::text_index::FullTextIndex,
        query_tokens: &[u32],
        avgdl: f32,
        hw_counter: &HardwareCounterCell,
        k1: f32,
        b: f32,
    ) -> OperationResult<f32> {
        let payload = self.payload_by_offset(point_id, hw_counter)?;
        let values = payload.get_value(field);
        if values.is_empty() {
            return Ok(0.0);
        }

        let mut tf: HashMap<u32, usize> = query_tokens
            .iter()
            .copied()
            .map(|token| (token, 0))
            .collect();
        let mut doc_len = 0usize;

        for value in values {
            match value {
                serde_json::Value::String(string) => {
                    text_index.tokenize_document_text(&string, |token| {
                        doc_len += 1;
                        if let Some(token_id) = text_index.token_id(token.as_ref(), hw_counter)
                            && let Some(count) = tf.get_mut(&token_id)
                        {
                            *count += 1;
                        }
                    });
                }
                serde_json::Value::Array(values) => {
                    for value in values {
                        if let serde_json::Value::String(string) = value {
                            text_index.tokenize_document_text(&string, |token| {
                                doc_len += 1;
                                if let Some(token_id) =
                                    text_index.token_id(token.as_ref(), hw_counter)
                                    && let Some(count) = tf.get_mut(&token_id)
                                {
                                    *count += 1;
                                }
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        if doc_len == 0 {
            return Ok(0.0);
        }

        let points_count = text_index.indexed_points_count() as f32;
        let doc_len = doc_len as f32;

        let mut score = 0.0;
        for token_id in query_tokens {
            let term_freq = tf.get(token_id).copied().unwrap_or(0) as f32;
            if term_freq == 0.0 {
                continue;
            }

            let doc_freq = text_index
                .get_posting_len(*token_id, hw_counter)
                .unwrap_or(0) as f32;
            if doc_freq <= 0.0 {
                continue;
            }

            let idf = ((points_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
            let norm = k1 * (1.0 - b + b * doc_len / avgdl);
            let tf_norm = term_freq * (k1 + 1.0) / (term_freq + norm);
            score += idf * tf_norm;
        }

        Ok(score)
    }
}
