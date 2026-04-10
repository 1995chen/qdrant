#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use common::counter::hardware_counter::HardwareCounterCell;
    use quantization::encoded_storage::{TestEncodedStorage, TestEncodedStorageBuilder};
    use quantization::encoded_vectors::{DistanceType, EncodedVectors, VectorParameters};
    use quantization::encoded_vectors_tq::{
        DEFAULT_TURBO_QUANT_BITS, EncodedVectorsTQ, TqCorrection, TqRotation,
    };
    use rand::{RngExt, SeedableRng};
    use tempfile::Builder;

    use crate::metrics::{dot_similarity, l1_similarity, l2_similarity};

    const VECTORS_COUNT: usize = 64;
    const VECTOR_DIM: usize = 768;
    const ERROR: f32 = VECTOR_DIM as f32 * 0.05;

    fn random_vectors(seed: u64, count: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..count)
            .map(|_| (0..dim).map(|_| rng.random()).collect())
            .collect()
    }

    fn random_vector(seed: u64, dim: usize) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..dim).map(|_| rng.random()).collect()
    }

    #[test]
    fn test_tq_dot() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<_>> = vec![];
        for _ in 0..VECTORS_COUNT {
            vector_data.push((0..VECTOR_DIM).map(|_| rng.random()).collect());
        }
        let query: Vec<_> = (0..VECTOR_DIM).map(|_| rng.random()).collect();

        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::Dot,
            invert: false,
        };
        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            VECTORS_COUNT,
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::default(),
            TqRotation::Random,
            None,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        let counter = HardwareCounterCell::new();
        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32, &counter);
            let orginal_score = dot_similarity(&query, vector);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_tq_l2() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<_>> = vec![];
        for _ in 0..VECTORS_COUNT {
            vector_data.push((0..VECTOR_DIM).map(|_| rng.random()).collect());
        }
        let query: Vec<_> = (0..VECTOR_DIM).map(|_| rng.random()).collect();

        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::L2,
            invert: false,
        };
        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            VECTORS_COUNT,
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::default(),
            TqRotation::Random,
            None,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        let counter = HardwareCounterCell::new();
        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32, &counter);
            let orginal_score = l2_similarity(&query, vector);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_tq_l1() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<_>> = vec![];
        for _ in 0..VECTORS_COUNT {
            vector_data.push((0..VECTOR_DIM).map(|_| rng.random()).collect());
        }
        let query: Vec<_> = (0..VECTOR_DIM).map(|_| rng.random()).collect();

        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::L1,
            invert: false,
        };
        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            VECTORS_COUNT,
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::default(),
            TqRotation::Random,
            None,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        let counter = HardwareCounterCell::new();
        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32, &counter);
            let orginal_score = l1_similarity(&query, vector);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_tq_dot_inverted() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<_>> = vec![];
        for _ in 0..VECTORS_COUNT {
            vector_data.push((0..VECTOR_DIM).map(|_| rng.random()).collect());
        }
        let query: Vec<_> = (0..VECTOR_DIM).map(|_| rng.random()).collect();

        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::Dot,
            invert: true,
        };
        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            VECTORS_COUNT,
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::default(),
            TqRotation::Random,
            None,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        let counter = HardwareCounterCell::new();
        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32, &counter);
            let orginal_score = -dot_similarity(&query, vector);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_tq_l2_inverted() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<_>> = vec![];
        for _ in 0..VECTORS_COUNT {
            vector_data.push((0..VECTOR_DIM).map(|_| rng.random()).collect());
        }
        let query: Vec<_> = (0..VECTOR_DIM).map(|_| rng.random()).collect();

        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::L2,
            invert: true,
        };
        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            VECTORS_COUNT,
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::default(),
            TqRotation::Random,
            None,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        let counter = HardwareCounterCell::new();
        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32, &counter);
            let orginal_score = -l2_similarity(&query, vector);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_tq_l1_inverted() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<_>> = vec![];
        for _ in 0..VECTORS_COUNT {
            vector_data.push((0..VECTOR_DIM).map(|_| rng.random()).collect());
        }
        let query: Vec<_> = (0..VECTOR_DIM).map(|_| rng.random()).collect();

        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::L1,
            invert: true,
        };
        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            VECTORS_COUNT,
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::default(),
            TqRotation::Random,
            None,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();
        let query_u8 = encoded.encode_query(&query);

        let counter = HardwareCounterCell::new();
        for (index, vector) in vector_data.iter().enumerate() {
            let score = encoded.score_point(&query_u8, index as u32, &counter);
            let orginal_score = -l1_similarity(&query, vector);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_tq_dot_internal() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<_>> = vec![];
        for _ in 0..VECTORS_COUNT {
            vector_data.push((0..VECTOR_DIM).map(|_| rng.random()).collect());
        }

        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::Dot,
            invert: false,
        };
        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            VECTORS_COUNT,
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::default(),
            TqRotation::Random,
            None,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();

        let counter = HardwareCounterCell::new();
        for i in 1..VECTORS_COUNT {
            let score = encoded.score_internal(0, i as u32, &counter);
            let orginal_score = dot_similarity(&vector_data[0], &vector_data[i]);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_tq_dot_inverted_internal() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut vector_data: Vec<Vec<_>> = vec![];
        for _ in 0..VECTORS_COUNT {
            vector_data.push((0..VECTOR_DIM).map(|_| rng.random()).collect());
        }

        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::Dot,
            invert: true,
        };
        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            VECTORS_COUNT,
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::default(),
            TqRotation::Random,
            None,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();

        let counter = HardwareCounterCell::new();
        for i in 1..VECTORS_COUNT {
            let score = encoded.score_internal(0, i as u32, &counter);
            let orginal_score = -dot_similarity(&vector_data[0], &vector_data[i]);
            assert!((score - orginal_score).abs() < ERROR);
        }
    }

    #[test]
    fn test_tq_recall_exact_vector_as_top1_for_dot() {
        let vector_data = random_vectors(100, 16, VECTOR_DIM);

        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::Dot,
            invert: false,
        };

        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            vector_data.len(),
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::default(),
            TqRotation::Random,
            None,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();

        let counter = HardwareCounterCell::new();
        for (expected_idx, query) in vector_data.iter().enumerate() {
            let query_encoded = encoded.encode_query(query);
            let best_idx = vector_data
                .iter()
                .enumerate()
                .map(|(idx, _)| {
                    (
                        idx,
                        encoded.score_point(&query_encoded, idx as u32, &counter),
                    )
                })
                .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            assert_eq!(best_idx, expected_idx);
        }
    }

    #[test]
    fn test_tq_recall_nearest_neighbor_for_l2() {
        let vector_data = random_vectors(101, 16, VECTOR_DIM);
        let mut query = vector_data[7].clone();
        for (idx, value) in query.iter_mut().enumerate() {
            *value += ((idx % 7) as f32 - 3.0) * 1e-3;
        }

        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::L2,
            invert: false,
        };

        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            vector_data.len(),
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::default(),
            TqRotation::Random,
            None,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();

        let query_encoded = encoded.encode_query(&query);
        let counter = HardwareCounterCell::new();
        let best_idx = vector_data
            .iter()
            .enumerate()
            .map(|(idx, _)| {
                (
                    idx,
                    encoded.score_point(&query_encoded, idx as u32, &counter),
                )
            })
            .min_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        assert_eq!(best_idx, 7);
    }

    #[test]
    fn test_tq_encode_internal_vector_matches_self_recall() {
        let vector_data = random_vectors(102, 16, VECTOR_DIM);

        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::Dot,
            invert: false,
        };

        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            vector_data.len(),
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::default(),
            TqRotation::Random,
            None,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();

        let counter = HardwareCounterCell::new();
        for expected_idx in 0..vector_data.len() {
            let query = encoded.encode_internal_vector(expected_idx as u32).unwrap();
            let best_idx = vector_data
                .iter()
                .enumerate()
                .map(|(idx, _)| (idx, encoded.score_point(&query, idx as u32, &counter)))
                .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            assert_eq!(best_idx, expected_idx);
        }
    }

    #[test]
    fn test_tq_qjl_corrections_recall_self_as_top1() {
        let vector_data = random_vectors(103, 12, VECTOR_DIM);
        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::Dot,
            invert: false,
        };
        let counter = HardwareCounterCell::new();

        for correction in [TqCorrection::Qjl, TqCorrection::QjlNormalization] {
            let quantized_vector_size =
                EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                    &vector_parameters,
                    DEFAULT_TURBO_QUANT_BITS,
                );
            let encoded = EncodedVectorsTQ::encode(
                vector_data.iter(),
                TestEncodedStorageBuilder::new(None, quantized_vector_size),
                &vector_parameters,
                vector_data.len(),
                DEFAULT_TURBO_QUANT_BITS,
                correction,
                TqRotation::Random,
                None,
                None,
                &AtomicBool::new(false),
            )
            .unwrap();

            for (expected_idx, query) in vector_data.iter().enumerate() {
                let query_encoded = encoded.encode_query(query);
                let best_idx = vector_data
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| {
                        (
                            idx,
                            encoded.score_point(&query_encoded, idx as u32, &counter),
                        )
                    })
                    .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                assert_eq!(
                    best_idx, expected_idx,
                    "failed for correction {correction:?}"
                );
            }
        }
    }

    #[test]
    fn test_tq_load_preserves_scores() {
        let dir = Builder::new().prefix("tq-load").tempdir().unwrap();
        let data_path = dir.path().join("vectors.bin");
        let meta_path = dir.path().join("meta.json");

        let vector_data = random_vectors(104, 12, VECTOR_DIM);
        let query = random_vector(105, VECTOR_DIM);

        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::Dot,
            invert: false,
        };
        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(Some(&data_path), quantized_vector_size),
            &vector_parameters,
            vector_data.len(),
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::QjlNormalization,
            TqRotation::Random,
            None,
            Some(&meta_path),
            &AtomicBool::new(false),
        )
        .unwrap();

        let query_encoded = encoded.encode_query(&query);
        let counter = HardwareCounterCell::new();
        let before_scores: Vec<f32> = (0..vector_data.len())
            .map(|idx| encoded.score_point(&query_encoded, idx as u32, &counter))
            .collect();

        let storage = TestEncodedStorage::from_file(&data_path, quantized_vector_size).unwrap();
        let loaded = EncodedVectorsTQ::load(storage, &meta_path).unwrap();
        let loaded_query = loaded.encode_query(&query);
        let after_scores: Vec<f32> = (0..vector_data.len())
            .map(|idx| loaded.score_point(&loaded_query, idx as u32, &counter))
            .collect();

        assert_eq!(before_scores.len(), after_scores.len());
        for (before, after) in before_scores.iter().zip(after_scores) {
            assert!((before - after).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tq_hadamard_rotation_on_power_of_two_dim_recall_self_as_top1() {
        let hadamard_dim = 1024;
        let vector_data = random_vectors(106, 8, hadamard_dim);
        let vector_parameters = VectorParameters {
            dim: hadamard_dim,
            deprecated_count: None,
            distance_type: DistanceType::Dot,
            invert: false,
        };
        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            vector_data.len(),
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::Normalization,
            TqRotation::Hadamard,
            None,
            None,
            &AtomicBool::new(false),
        )
        .unwrap();

        let counter = HardwareCounterCell::new();
        for (expected_idx, query) in vector_data.iter().enumerate() {
            let query_encoded = encoded.encode_query(query);
            let best_idx = vector_data
                .iter()
                .enumerate()
                .map(|(idx, _)| {
                    (
                        idx,
                        encoded.score_point(&query_encoded, idx as u32, &counter),
                    )
                })
                .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            assert_eq!(best_idx, expected_idx);
        }
    }

    #[test]
    fn test_tq_hadamard_rotation_on_non_power_of_two_dim_recall_self_as_top1() {
        let vector_data = random_vectors(107, 8, VECTOR_DIM);
        let vector_parameters = VectorParameters {
            dim: VECTOR_DIM,
            deprecated_count: None,
            distance_type: DistanceType::Dot,
            invert: false,
        };
        let quantized_vector_size =
            EncodedVectorsTQ::<TestEncodedStorage>::get_quantized_vector_size(
                &vector_parameters,
                DEFAULT_TURBO_QUANT_BITS,
            );
        let encoded = EncodedVectorsTQ::encode(
            vector_data.iter(),
            TestEncodedStorageBuilder::new(None, quantized_vector_size),
            &vector_parameters,
            vector_data.len(),
            DEFAULT_TURBO_QUANT_BITS,
            TqCorrection::Normalization,
            TqRotation::Hadamard,
            Some(128),
            None,
            &AtomicBool::new(false),
        )
        .unwrap();

        let counter = HardwareCounterCell::new();
        for (expected_idx, query) in vector_data.iter().enumerate() {
            let query_encoded = encoded.encode_query(query);
            let best_idx = vector_data
                .iter()
                .enumerate()
                .map(|(idx, _)| {
                    (
                        idx,
                        encoded.score_point(&query_encoded, idx as u32, &counter),
                    )
                })
                .max_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            assert_eq!(best_idx, expected_idx);
        }
    }
}
