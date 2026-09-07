use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::counter::hardware_counter::HardwareCounterCell;
use itertools::Itertools;
use segment::types::Filter;
use shard::query::ShardQueryRequest;
use shard::query::payload_query::{
    TextQueryStats, TextQueryStatsRequest, validate_text_query_schema,
};
use shard::segment_holder::locked::LockedSegmentHolder;
use tokio_util::task::AbortOnDropHandle;

use super::LocalShard;
use crate::operations::types::{CollectionError, CollectionResult};

impl LocalShard {
    pub(super) async fn resolve_text_queries(
        &self,
        requests: &mut [ShardQueryRequest],
        timeout: Duration,
        hw_measurement_acc: HwMeasurementAcc,
        is_stopped: Arc<AtomicBool>,
    ) -> CollectionResult<()> {
        let started_at = std::time::Instant::now();
        let stats_requests = requests
            .iter()
            .flat_map(ShardQueryRequest::text_query_stats_requests)
            .unique()
            .collect_vec();

        for stats_request in stats_requests {
            let stats = self
                .text_query_stats(
                    stats_request.clone(),
                    timeout.saturating_sub(started_at.elapsed()),
                    hw_measurement_acc.clone(),
                    Arc::clone(&is_stopped),
                )
                .await?;
            let (weights, average_document_length) = stats.into_query_parts();
            for request in requests.iter_mut() {
                request.set_text_query_stats(&stats_request, &weights, average_document_length);
            }
        }

        Ok(())
    }

    async fn text_query_stats(
        &self,
        request: TextQueryStatsRequest,
        timeout: Duration,
        hw_measurement_acc: HwMeasurementAcc,
        is_stopped: Arc<AtomicBool>,
    ) -> CollectionResult<TextQueryStats> {
        let started_at = std::time::Instant::now();
        {
            let Some(schema) = self.payload_index_schema.try_read_for(timeout) else {
                return Err(CollectionError::timeout(
                    timeout,
                    "validate_text_query_schema",
                ));
            };
            validate_text_query_schema(&request.key, schema.schema.get(&request.key))?;
        }
        let remaining_timeout = timeout.saturating_sub(started_at.elapsed());
        if remaining_timeout.is_zero() {
            return Err(CollectionError::timeout(
                timeout,
                "compute_text_query_stats",
            ));
        }
        let segments = self.segments.clone();
        let hw_counter = hw_measurement_acc.get_counter_cell();
        let cpu_utilization = hw_measurement_acc.cpu_utilization();
        let task = AbortOnDropHandle::new(self.search_runtime.spawn_blocking(move || {
            cpu_utilization.measure(|| {
                Self::compute_text_query_stats(
                    segments,
                    &request.key,
                    &request.query_str,
                    request.corpus.as_ref(),
                    &hw_counter,
                    is_stopped.as_ref(),
                    remaining_timeout,
                )
            })
        }));
        tokio::time::timeout(remaining_timeout, task)
            .await
            .map_err(|_| CollectionError::timeout(timeout, "compute_text_query_stats"))??
    }

    fn compute_text_query_stats(
        segments: LockedSegmentHolder,
        key: &segment::json_path::JsonPath,
        query_str: &str,
        corpus: Option<&Filter>,
        hw_counter: &HardwareCounterCell,
        is_stopped: &AtomicBool,
        timeout: Duration,
    ) -> CollectionResult<TextQueryStats> {
        let started_at = std::time::Instant::now();
        let segments: Vec<_> = {
            let Some(segments_guard) = segments.try_read_for(timeout) else {
                return Err(CollectionError::timeout(
                    timeout,
                    "compute_text_query_stats",
                ));
            };
            segments_guard
                .non_appendable_then_appendable_segments()
                .collect()
        };
        let mut merged: Option<TextQueryStats> = None;
        for segment in segments {
            if is_stopped.load(Ordering::Relaxed) {
                return Err(CollectionError::cancelled(
                    "Text query statistics collection was cancelled",
                ));
            }
            let segment = segment.get();
            let remaining_timeout = timeout.saturating_sub(started_at.elapsed());
            let Some(segment_guard) = segment.try_read_for(remaining_timeout) else {
                return Err(CollectionError::timeout(
                    timeout,
                    "compute_text_query_stats",
                ));
            };
            if is_stopped.load(Ordering::Relaxed) {
                return Err(CollectionError::cancelled(
                    "Text query statistics collection was cancelled",
                ));
            }
            let stats =
                segment_guard.payload_text_stats(key, query_str, corpus, is_stopped, hw_counter)?;
            let stats = TextQueryStats::from(stats);
            if let Some(merged) = &mut merged {
                merged.merge(stats)?;
            } else {
                merged = Some(stats);
            }
        }
        Ok(merged.unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use segment::json_path::JsonPath;
    use shard::segment_holder::SegmentHolder;

    use super::*;

    #[test]
    fn text_query_stats_honors_segment_holder_timeout() {
        let segments = LockedSegmentHolder::new(SegmentHolder::default());
        let _write_guard = segments.write();

        let result = LocalShard::compute_text_query_stats(
            segments.clone(),
            &JsonPath::new("text"),
            "query",
            None,
            &HardwareCounterCell::disposable(),
            &AtomicBool::new(false),
            Duration::ZERO,
        );

        assert!(matches!(result, Err(CollectionError::Timeout { .. })));
    }

    #[test]
    fn text_query_stats_honors_segment_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let segment = shard::fixtures::empty_segment(dir.path());
        let locked_segment = shard::locked_segment::LockedSegment::new(segment);
        let segments = LockedSegmentHolder::new(SegmentHolder::default());
        segments.write().add_new(locked_segment.clone());
        let _write_guard = locked_segment.get().write();

        let result = LocalShard::compute_text_query_stats(
            segments,
            &JsonPath::new("text"),
            "query",
            None,
            &HardwareCounterCell::disposable(),
            &AtomicBool::new(false),
            Duration::ZERO,
        );

        assert!(matches!(result, Err(CollectionError::Timeout { .. })));
    }
}
