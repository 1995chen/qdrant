#[cfg(test)]
mod tests {
    use quantization::turboquant::{
        NormCorrection, RotationKind, TurboQuantCodec, TurboQuantConfig, evaluate_recall,
    };
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn sample_standard_normal(rng: &mut StdRng) -> f32 {
        let u1 = (1.0f32 - rng.random::<f32>()).max(1e-12f32);
        let u2 = rng.random::<f32>();
        let radius = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * std::f32::consts::PI * u2;
        radius * theta.cos()
    }

    fn normalize(values: &mut [f32]) {
        let norm = values.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            values.iter_mut().for_each(|value| *value /= norm);
        }
    }

    fn synthetic_dataset(
        seed: u64,
        vector_count: usize,
        query_count: usize,
        dim: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut rng = StdRng::seed_from_u64(seed);

        let cluster_count = 12usize;
        let mut centers = Vec::with_capacity(cluster_count);
        for _ in 0..cluster_count {
            let mut center: Vec<f32> = (0..dim).map(|_| sample_standard_normal(&mut rng)).collect();
            normalize(&mut center);
            centers.push(center);
        }

        let mut dataset = Vec::with_capacity(vector_count);
        for _ in 0..vector_count {
            let cluster = rng.random_range(0..cluster_count);
            let norm_scale = rng.random_range(0.35f32..2.2f32);
            let vector: Vec<f32> = centers[cluster]
                .iter()
                .map(|&value| {
                    let noise = 0.09 * sample_standard_normal(&mut rng);
                    norm_scale * (value + noise)
                })
                .collect();
            dataset.push(vector);
        }

        let mut queries = Vec::with_capacity(query_count);
        for _ in 0..query_count {
            let source_index = rng.random_range(0..dataset.len());
            let query: Vec<f32> = dataset[source_index]
                .iter()
                .map(|&value| value + 0.04 * sample_standard_normal(&mut rng))
                .collect();
            queries.push(query);
        }

        (dataset, queries)
    }

    #[test]
    fn turboquant_pack_unpack_roundtrip_for_three_bits() {
        let config = TurboQuantConfig::baseline(64, 3, 7);
        let codec = TurboQuantCodec::new(config).unwrap();

        let vector: Vec<f32> = (0..64).map(|index| (index as f32 * 0.1).sin()).collect();
        let encoded = codec.quantize(&vector).unwrap();
        let decoded = codec.dequantize(&encoded);

        assert_eq!(encoded.packed_levels.len(), (64usize * 3usize).div_ceil(8));
        assert_eq!(decoded.len(), 64);
    }

    #[test]
    fn baseline_turboquant_recall_is_reasonable() {
        let (dataset, queries) = synthetic_dataset(41, 768, 64, 64);
        let codec = TurboQuantCodec::new(TurboQuantConfig::baseline(64, 3, 41)).unwrap();
        let encoded = codec.quantize_batch(&dataset).unwrap();

        let report = evaluate_recall(&codec, &encoded, &dataset, &queries, &[10, 100], false);

        eprintln!("baseline report: {report:?}");
        assert!(report.recall(10).unwrap() > 0.82);
        assert!(report.recall(100).unwrap() > 0.92);
    }

    #[test]
    fn qjl_variant_keeps_useful_recall() {
        let (dataset, queries) = synthetic_dataset(77, 768, 64, 64);
        let codec = TurboQuantCodec::new(TurboQuantConfig::with_qjl(64, 3, 77)).unwrap();
        let encoded = codec.quantize_batch(&dataset).unwrap();

        let report = evaluate_recall(&codec, &encoded, &dataset, &queries, &[10, 100], false);

        eprintln!("qjl report: {report:?}");
        assert!(report.recall(10).unwrap() > 0.74);
        assert!(report.recall(100).unwrap() > 0.89);
    }

    #[test]
    fn norm_correction_improves_three_bit_recall_on_average() {
        let mut baseline_r10 = 0.0;
        let mut corrected_r10 = 0.0;
        let mut baseline_r100 = 0.0;
        let mut corrected_r100 = 0.0;

        for seed in [5_u64, 11, 19, 29] {
            let (dataset, queries) = synthetic_dataset(seed, 768, 64, 64);

            let baseline = TurboQuantCodec::new(TurboQuantConfig {
                dim: 64,
                bit_width: 3,
                rotation: RotationKind::Haar,
                seed,
                qjl: false,
                norm_correction: NormCorrection::Disabled,
            })
            .unwrap();
            let corrected = TurboQuantCodec::new(TurboQuantConfig {
                dim: 64,
                bit_width: 3,
                rotation: RotationKind::Haar,
                seed,
                qjl: false,
                norm_correction: NormCorrection::Exact,
            })
            .unwrap();

            let baseline_encoded = baseline.quantize_batch(&dataset).unwrap();
            let corrected_encoded = corrected.quantize_batch(&dataset).unwrap();

            let baseline_report = evaluate_recall(
                &baseline,
                &baseline_encoded,
                &dataset,
                &queries,
                &[10, 100],
                false,
            );
            let corrected_report = evaluate_recall(
                &corrected,
                &corrected_encoded,
                &dataset,
                &queries,
                &[10, 100],
                false,
            );

            baseline_r10 += baseline_report.recall(10).unwrap();
            corrected_r10 += corrected_report.recall(10).unwrap();
            baseline_r100 += baseline_report.recall(100).unwrap();
            corrected_r100 += corrected_report.recall(100).unwrap();
        }

        baseline_r10 /= 4.0;
        corrected_r10 /= 4.0;
        baseline_r100 /= 4.0;
        corrected_r100 /= 4.0;

        eprintln!(
            "norm correction avg r10: baseline={baseline_r10:.4}, corrected={corrected_r10:.4}; \
             r100: baseline={baseline_r100:.4}, corrected={corrected_r100:.4}"
        );

        assert!(corrected_r10 >= baseline_r10 + 0.01);
        assert!(corrected_r100 >= baseline_r100);
    }

    #[test]
    fn simd_scores_match_plain_scores() {
        let (dataset, queries) = synthetic_dataset(123, 192, 8, 64);
        let codec = TurboQuantCodec::new(TurboQuantConfig {
            dim: 64,
            bit_width: 4,
            rotation: RotationKind::Hadamard,
            seed: 123,
            qjl: false,
            norm_correction: NormCorrection::Exact,
        })
        .unwrap();
        let encoded = codec.quantize_batch(&dataset).unwrap();

        for query in &queries {
            for vector in &encoded {
                let plain = codec.score_dot_plain(query, vector);
                let simd = codec.score_dot_simd(query, vector);
                assert!((plain - simd).abs() < 1e-4);
            }
        }

        let plain_report =
            evaluate_recall(&codec, &encoded, &dataset, &queries, &[10, 100], false);
        let simd_report = evaluate_recall(&codec, &encoded, &dataset, &queries, &[10, 100], true);
        assert_eq!(plain_report, simd_report);
    }
}
