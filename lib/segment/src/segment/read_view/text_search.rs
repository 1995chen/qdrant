use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use common::bitvec::{BitSliceExt as _, BitVec};
use common::condition_checker::ConditionChecker;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;

use crate::common::check_stopped;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::query_context::{PayloadTextIndexStats, PayloadTextSearchContext};
use crate::id_tracker::IdTrackerRead;
use crate::index::PayloadIndexRead;
use crate::index::field_index::full_text_index::full_text_index_read::FullTextIndexRead;
use crate::index::field_index::full_text_index::full_text_index_scoring::FullTextIndexScoring;
use crate::index::field_index::full_text_index::{ParsedQuery, TokenSet};
use crate::payload_storage::PayloadStorageRead;
use crate::segment::read_view::SegmentReadView;
use crate::segment::vector_data_read::VectorDataRead;
use crate::types::{
    DEFAULT_SPARSE_FULL_SCAN_THRESHOLD, Filter, ScoredPoint, WithPayload, WithVector,
};

enum CorpusPoints {
    SortedIds(Vec<PointOffsetType>),
    Mask(BitVec),
}

impl<'s, TIdT, TPI, TPS, TVD> SegmentReadView<'s, TIdT, TPI, TPS, TVD>
where
    TIdT: IdTrackerRead,
    TPI: PayloadIndexRead,
    TPS: PayloadStorageRead,
    TVD: VectorDataRead,
{
    pub fn search_payload_text(
        &self,
        ctx: Arc<PayloadTextSearchContext>,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<Vec<ScoredPoint>> {
        let PayloadTextSearchContext {
            key,
            query,
            filter,
            top,
            is_stopped,
        } = &*ctx;

        if *top == 0 || query.query_tokens().is_empty() {
            return Ok(vec![]);
        }

        check_stopped(is_stopped)?;

        let Some(text_index) = self.payload_index.full_text_index_for(key) else {
            return Err(OperationError::validation_error(format!(
                "payload field `{key}` does not have a full-text index",
            )));
        };

        let use_plain_filtered_search = match filter {
            Some(filter) => {
                let query_cardinality = self.estimate_point_count(Some(filter), hw_counter)?;
                query_cardinality.max < DEFAULT_SPARSE_FULL_SCAN_THRESHOLD
            }
            None => false,
        };

        let internal_results = if let Some(filter) = filter {
            if use_plain_filtered_search {
                let mut prefiltered_points =
                    self.payload_index
                        .query_points(filter, hw_counter, is_stopped.as_ref())?;
                prefiltered_points.sort_unstable();
                text_index.search_text_index_plain(
                    query,
                    *top,
                    &prefiltered_points,
                    is_stopped.as_ref(),
                )?
            } else {
                let filter_context = self.payload_index.filter_context(filter, hw_counter)?;
                text_index.search_text_index(query, *top, is_stopped.as_ref(), |point_id| {
                    self.payload_text_point_is_visible(point_id)
                        && filter_context.check_infallible(point_id)
                })?
            }
        } else {
            text_index.search_text_index(query, *top, is_stopped.as_ref(), |point_id| {
                self.payload_text_point_is_visible(point_id)
            })?
        };

        self.process_search_result(
            internal_results,
            &WithPayload::from(false),
            &WithVector::from(false),
            hw_counter,
            is_stopped,
        )
    }

    pub fn payload_text_stats(
        &self,
        key: &crate::json_path::JsonPath,
        query_str: &str,
        corpus: Option<&Filter>,
        is_stopped: &AtomicBool,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<PayloadTextIndexStats> {
        let Some(text_index) = self.payload_index.full_text_index_for(key) else {
            return Ok(PayloadTextIndexStats::default());
        };

        let mut stats = PayloadTextIndexStats::new(text_index.tokenize_query_str(query_str));

        let mut token_ids = vec![None; stats.tokens.len()];
        let iter = stats
            .tokens
            .iter()
            .enumerate()
            .map(|(idx, token)| (idx, token.as_str()));
        text_index.for_each_token_id(iter, hw_counter, |idx, token_id| {
            token_ids[idx] = token_id;
        })?;

        self.populate_global_text_stats(text_index, &mut stats, is_stopped, hw_counter)?;

        if corpus.is_none() {
            stats.document_count = stats.global_document_count;
            stats.sum_document_length = stats.global_sum_document_length;
            // Posting iterators avoid the previous full-corpus point scan.
            // They already exclude deletions known to the field index; the ID
            // tracker predicate also excludes tombstones left behind by
            // append-only updates and points hidden behind a deferred cutoff.
            for (idx, token_id) in token_ids.into_iter().enumerate() {
                check_stopped(is_stopped)?;
                let Some(token_id) = token_id else {
                    continue;
                };
                let query = ParsedQuery::AllTokens(TokenSet::from_iter([token_id]));
                for (posting_index, point_id) in
                    text_index.filter_query(query, hw_counter)?.enumerate()
                {
                    if posting_index.is_multiple_of(1024) {
                        check_stopped(is_stopped)?;
                    }
                    if self.payload_text_point_is_visible(point_id) {
                        stats.document_frequencies[idx] += 1;
                    }
                }
            }
            check_stopped(is_stopped)?;
            return Ok(stats);
        }

        // Explicit IDF corpora remain filter-aware. SearchParams.idf.corpus
        // scopes N and df only; average document length above always comes
        // from all visible documents in the segment.
        // A proxy segment always supplies a corpus after adding its overlay
        // exclusions, so it deliberately stays on this path.
        let corpus = corpus.expect("checked above");

        let (corpus_doc_count, corpus_document_length, corpus_points) =
            self.collect_text_corpus_points(text_index, corpus, is_stopped, hw_counter)?;
        stats.document_count += corpus_doc_count;
        stats.sum_document_length += corpus_document_length;
        match corpus_points {
            CorpusPoints::SortedIds(corpus_ids) => {
                for (idx, token_id) in token_ids.into_iter().enumerate() {
                    check_stopped(is_stopped)?;
                    let Some(token_id) = token_id else {
                        continue;
                    };
                    let query = ParsedQuery::AllTokens(TokenSet::from_iter([token_id]));
                    for (corpus_index, &point_id) in corpus_ids.iter().enumerate() {
                        if corpus_index.is_multiple_of(1024) {
                            check_stopped(is_stopped)?;
                        }
                        if text_index.check_match(&query, point_id)? {
                            stats.document_frequencies[idx] += 1;
                        }
                    }
                }
            }
            CorpusPoints::Mask(mask) => {
                for (idx, token_id) in token_ids.into_iter().enumerate() {
                    check_stopped(is_stopped)?;
                    let Some(token_id) = token_id else {
                        continue;
                    };
                    let query = ParsedQuery::AllTokens(TokenSet::from_iter([token_id]));
                    for (posting_index, point_id) in
                        text_index.filter_query(query, hw_counter)?.enumerate()
                    {
                        if posting_index.is_multiple_of(1024) {
                            check_stopped(is_stopped)?;
                        }
                        if mask.get_bit(point_id as usize).unwrap_or(false) {
                            stats.document_frequencies[idx] += 1;
                        }
                    }
                }
            }
        }
        check_stopped(is_stopped)?;
        Ok(stats)
    }

    /// Whether an indexed payload document belongs to the visible read view.
    ///
    /// Full-text indexes do not own deferred-point semantics, so both search
    /// candidates and statistics must apply the ID tracker's cutoff here.
    fn payload_text_point_is_visible(&self, point_id: PointOffsetType) -> bool {
        !self.id_tracker.is_deleted_point(point_id)
            && self
                .id_tracker
                .deferred_internal_id()
                .is_none_or(|cutoff| point_id < cutoff)
    }

    fn populate_global_text_stats(
        &self,
        text_index: &impl FullTextIndexRead,
        stats: &mut PayloadTextIndexStats,
        is_stopped: &AtomicBool,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        let Some((document_count, sum_document_length)) = text_index.bm25_document_stats() else {
            // A valid payload BM25 query always has aggregate state. Keep a
            // compatibility fallback for older/non-BM25 indexes so this read
            // surface does not change their error timing.
            let global_corpus = Filter::default();
            let (document_count, sum_document_length, _) = self.collect_text_corpus_points(
                text_index,
                &global_corpus,
                is_stopped,
                hw_counter,
            )?;
            stats.global_document_count = document_count;
            stats.global_sum_document_length = sum_document_length;
            return Ok(());
        };
        stats.global_document_count = document_count;
        stats.global_sum_document_length = sum_document_length;
        self.remove_invisible_documents_from_global_text_stats(
            text_index, stats, is_stopped, hw_counter,
        )
    }

    /// Remove documents present in the full-text index aggregate but hidden by
    /// the ID tracker. This is proportional to tombstones plus the deferred
    /// tail, rather than to every live point in the segment.
    fn remove_invisible_documents_from_global_text_stats(
        &self,
        text_index: &impl FullTextIndexRead,
        stats: &mut PayloadTextIndexStats,
        is_stopped: &AtomicBool,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        let total_points = self.id_tracker.total_point_count();
        let visible_end = self
            .id_tracker
            .deferred_internal_id()
            .map_or(total_points, |cutoff| (cutoff as usize).min(total_points));

        let deleted = self.id_tracker.deleted_point_bitslice();
        for (deleted_index, point_id) in deleted[..deleted.len().min(visible_end)]
            .iter_ones()
            .enumerate()
        {
            if deleted_index.is_multiple_of(1024) {
                check_stopped(is_stopped)?;
            }
            Self::remove_indexed_document_from_text_stats(
                text_index,
                point_id as PointOffsetType,
                stats,
                hw_counter,
            )?;
        }

        for point_id in visible_end..total_points {
            if point_id.is_multiple_of(1024) {
                check_stopped(is_stopped)?;
            }
            Self::remove_indexed_document_from_text_stats(
                text_index,
                point_id as PointOffsetType,
                stats,
                hw_counter,
            )?;
        }
        Ok(())
    }

    fn remove_indexed_document_from_text_stats(
        text_index: &impl FullTextIndexRead,
        point_id: PointOffsetType,
        stats: &mut PayloadTextIndexStats,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<()> {
        // Normal payload deletion removes the document from the field index
        // and its aggregate already. Append-only tombstones intentionally do
        // not; only those still-indexed documents need subtracting here.
        if text_index.values_is_empty(point_id) {
            return Ok(());
        }
        let Some(document_length) = text_index.document_length(point_id, hw_counter)? else {
            return Err(OperationError::service_error(
                "BM25 document length is missing for an indexed payload document",
            ));
        };
        stats.global_document_count =
            stats.global_document_count.checked_sub(1).ok_or_else(|| {
                OperationError::service_error("BM25 document count aggregate is inconsistent")
            })?;
        stats.global_sum_document_length = stats
            .global_sum_document_length
            .checked_sub(u64::from(document_length))
            .ok_or_else(|| {
                OperationError::service_error("BM25 document length aggregate is inconsistent")
            })?;
        Ok(())
    }

    fn collect_text_corpus_points(
        &self,
        text_index: &impl FullTextIndexRead,
        corpus: &Filter,
        is_stopped: &AtomicBool,
        hw_counter: &HardwareCounterCell,
    ) -> OperationResult<(usize, u64, CorpusPoints)> {
        let total_points = self.id_tracker.total_point_count();
        let id_list_threshold = (total_points / 32).max(128);
        let corpus_points = self
            .payload_index
            .query_points(corpus, hw_counter, is_stopped)?;

        let mut corpus_ids = Vec::new();
        for (corpus_index, point_id) in corpus_points.into_iter().enumerate() {
            if corpus_index.is_multiple_of(1024) {
                check_stopped(is_stopped)?;
            }
            if !text_index.values_is_empty(point_id) {
                corpus_ids.push(point_id);
            }
        }
        check_stopped(is_stopped)?;

        let document_count = corpus_ids.len();
        let mut sum_document_length = 0u64;
        for (corpus_index, &point_id) in corpus_ids.iter().enumerate() {
            if corpus_index.is_multiple_of(1024) {
                check_stopped(is_stopped)?;
            }
            let Some(document_length) = text_index.document_length(point_id, hw_counter)? else {
                return Err(OperationError::service_error(
                    "BM25 document length is missing for an indexed payload document",
                ));
            };
            sum_document_length = sum_document_length.saturating_add(u64::from(document_length));
        }
        let corpus_points = if document_count > id_list_threshold {
            let mut mask = BitVec::repeat(false, total_points);
            for (corpus_index, point_id) in corpus_ids.into_iter().enumerate() {
                if corpus_index.is_multiple_of(1024) {
                    check_stopped(is_stopped)?;
                }
                mask.set(point_id as usize, true);
            }
            CorpusPoints::Mask(mask)
        } else {
            corpus_ids.sort_unstable();
            CorpusPoints::SortedIds(corpus_ids)
        };
        Ok((document_count, sum_document_length, corpus_points))
    }
}
