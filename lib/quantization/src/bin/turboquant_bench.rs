//! Standalone TurboQuant benchmark CLI.
//!
//! The goal of this binary is to make algorithm iteration fast while the
//! implementation is still isolated from the rest of Qdrant. We deliberately
//! avoid external CLI dependencies here so the tool stays lightweight.
//!
//! Example:
//! `cargo run -p quantization --bin turboquant_bench -- --dataset-path ./dataset --vectors 2000 --queries 200 --bits 3`

use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use arrow_array::{Array, Float32Array, Float64Array, GenericListArray, OffsetSizeTrait};
use arrow_ipc::reader::{FileReader, StreamReader};
use quantization::turboquant::{
    NormCorrection, RecallReport, RotationKind, TurboQuantCodec, TurboQuantConfig,
    compute_exact_baseline, evaluate_recall_with_baseline,
};

#[derive(Debug, Clone)]
struct Args {
    vectors: usize,
    queries: usize,
    bits: u8,
    dataset_path: Option<PathBuf>,
    dataset_column: String,
    query_offset: Option<usize>,
    help: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            vectors: 2000,
            queries: 200,
            bits: 3,
            dataset_path: None,
            dataset_column: "openai".to_owned(),
            query_offset: None,
            help: false,
        }
    }
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut parsed = Self::default();
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--vectors" => parsed.vectors = parse_value(&arg, args.next())?,
                "--queries" => parsed.queries = parse_value(&arg, args.next())?,
                "--bits" => parsed.bits = parse_value(&arg, args.next())?,
                "--dataset-path" => {
                    parsed.dataset_path = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| format!("missing value after `{arg}`"))?,
                    ));
                }
                "--dataset-column" => {
                    parsed.dataset_column = args
                        .next()
                        .ok_or_else(|| format!("missing value after `{arg}`"))?;
                }
                "--query-offset" => parsed.query_offset = Some(parse_value(&arg, args.next())?),
                "-h" | "--help" => parsed.help = true,
                other => return Err(format!("unknown argument `{other}`")),
            }
        }
        Ok(parsed)
    }
}

struct Variant {
    name: String,
    config: TurboQuantConfig,
    use_simd: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("turboquant_bench error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse(std::env::args().skip(1))?;
    if args.help {
        println!(
            "Usage:
  cargo run -p quantization --bin turboquant_bench -- [options]

Options:
  --dataset-path <path>    Hugging Face Arrow cache directory or one `.arrow` shard
  --dataset-column <name>  Embedding column to read. Default: openai
  --vectors <usize>     Database vector count. Default: 2000
  --queries <usize>     Query count. Default: 200
  --bits <u8>           Scalar quantization bits. Default: 3
  --query-offset <usize>
                        Query slice start row inside the dataset. Default: same as --vectors
  -h, --help            Show this help"
        );
        return Ok(());
    }

    let dataset_path = args
        .dataset_path
        .as_ref()
        .ok_or_else(|| "--dataset-path is required".to_owned())?
        .clone();
    let (dataset, queries, dim) = load_arrow_dataset(
        &dataset_path,
        args.dataset_column,
        args.vectors,
        args.queries,
        args.query_offset.unwrap_or(args.vectors),
    )?;
    let ks = [10, 100];
    let seed = 42;
    let variants = build_variants(dim, args.bits, 42);
    let exact_baseline = compute_exact_baseline(&dataset, &queries, &ks);

    println!(
        "TurboQuant benchmark\n  dim={} vectors={} queries={} bits={} seed={} source={}\n",
        dim,
        args.vectors,
        args.queries,
        args.bits,
        seed,
        dataset_path.display()
    );
    println!(
        "{:<24} {:>10} {:>10} {:>11} {:>11} {:>11} {:>12} {:>12}",
        "variant",
        "recall@10",
        "recall@100",
        "encode_ms",
        "search_ms",
        "total_ms",
        "dim",
        "qjl_bits"
    );
    println!("{}", "-".repeat(108));
    print_variant_row(
        "exact/f32",
        &exact_baseline.report,
        None,
        exact_baseline.elapsed,
        exact_baseline.elapsed,
        dim,
        0,
    );

    for variant in variants {
        let total_started = Instant::now();
        let codec = TurboQuantCodec::new(variant.config.clone()).map_err(|err| err.to_string())?;
        let encode_started = Instant::now();
        let encoded = codec
            .quantize_batch(&dataset)
            .map_err(|err| err.to_string())?;
        let encode_elapsed = encode_started.elapsed();
        let evaluation = evaluate_recall_with_baseline(
            &codec,
            &encoded,
            &queries,
            &ks,
            variant.use_simd,
            &exact_baseline,
        );
        let total_elapsed = total_started.elapsed();
        print_variant_row(
            &variant.name,
            &evaluation.report,
            Some(encode_elapsed),
            evaluation.elapsed,
            total_elapsed,
            dim,
            if variant.config.qjl { dim } else { 0 },
        );
    }

    Ok(())
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn print_variant_row(
    name: &str,
    report: &RecallReport,
    encode_elapsed: Option<Duration>,
    search_elapsed: Duration,
    total_elapsed: Duration,
    dim: usize,
    qjl_bits: usize,
) {
    let recall_10 = report.recall(10).unwrap_or(0.0);
    let recall_100 = report.recall(100).unwrap_or(0.0);
    let encode_ms = encode_elapsed
        .map(duration_ms)
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "-".to_owned());
    println!(
        "{:<24} {:>10.4} {:>10.4} {:>11} {:>11.3} {:>11.3} {:>12} {:>12}",
        name,
        recall_10,
        recall_100,
        encode_ms,
        duration_ms(search_elapsed),
        duration_ms(total_elapsed),
        dim,
        qjl_bits
    );
}

fn build_variants(dim: usize, bits: u8, seed: u64) -> Vec<Variant> {
    let scalar = Variant {
        name: "scalar/DenseHaar".into(),
        config: TurboQuantConfig {
            dim,
            bit_width: bits,
            rotation: RotationKind::DenseHaar,
            seed: seed,
            qjl: false,
            norm_correction: NormCorrection::Disabled,
        },
        use_simd: false,
    };

    let qjl_dense = Variant {
        name: "qjl/DenseHaar".into(),
        config: TurboQuantConfig {
            dim,
            bit_width: bits,
            rotation: RotationKind::DenseHaar,
            seed: seed,
            qjl: true,
            norm_correction: NormCorrection::Disabled,
        },
        use_simd: false,
    };

    let qjl_norm_dense = Variant {
        name: "qjl+norm/DenseHaar".into(),
        config: TurboQuantConfig {
            dim,
            bit_width: bits,
            rotation: RotationKind::DenseHaar,
            seed: seed,
            qjl: true,
            norm_correction: NormCorrection::Exact,
        },
        use_simd: false,
    };

    let qjl_norm_wht = Variant {
        name: "qjl+norm/WalshHadamard".into(),
        config: TurboQuantConfig {
            dim,
            bit_width: bits,
            rotation: RotationKind::WalshHadamard,
            seed: seed,
            qjl: true,
            norm_correction: NormCorrection::Exact,
        },
        use_simd: false,
    };

    let norm = Variant {
        name: "norm/DenseHaar".into(),
        config: TurboQuantConfig {
            dim,
            bit_width: bits,
            rotation: RotationKind::DenseHaar,
            seed: seed,
            qjl: false,
            norm_correction: NormCorrection::Exact,
        },
        use_simd: false,
    };

    let simd = Variant {
        name: "simd/DenseHaar".into(),
        config: TurboQuantConfig {
            dim,
            bit_width: bits,
            rotation: RotationKind::DenseHaar,
            seed: seed,
            qjl: false,
            norm_correction: NormCorrection::Exact,
        },
        use_simd: true,
    };

    vec![scalar, qjl_dense, qjl_norm_dense, qjl_norm_wht, norm, simd]
}

fn parse_value<T: std::str::FromStr>(flag: &str, value: Option<String>) -> Result<T, String> {
    let value = value.ok_or_else(|| format!("missing value after `{flag}`"))?;
    value
        .parse()
        .map_err(|_| format!("failed to parse value `{value}` for `{flag}`"))
}

fn load_arrow_dataset(
    path: &Path,
    column_name: String,
    vectors: usize,
    queries: usize,
    query_offset: usize,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, usize), String> {
    let needed_rows = query_offset
        .checked_add(queries)
        .ok_or_else(|| "query window overflow".to_owned())?
        .max(vectors);
    let mut rows = Vec::with_capacity(needed_rows);
    let mut inferred_dim = None;

    for file_path in collect_arrow_files(path)? {
        if rows.len() >= needed_rows {
            break;
        }
        let file = File::open(&file_path)
            .map_err(|err| format!("failed to open {}: {err}", file_path.display()))?;
        if let Ok(reader) = FileReader::try_new(file, None) {
            append_record_batches(
                reader,
                "Arrow file",
                &file_path,
                &column_name,
                needed_rows,
                &mut rows,
                &mut inferred_dim,
            )?;
        } else {
            let stream = StreamReader::try_new(
                BufReader::new(
                    File::open(&file_path)
                        .map_err(|err| format!("failed to open {}: {err}", file_path.display()))?,
                ),
                None,
            )
            .map_err(|err| format!("failed to read Arrow stream {}: {err}", file_path.display()))?;
            append_record_batches(
                stream,
                "Arrow stream",
                &file_path,
                &column_name,
                needed_rows,
                &mut rows,
                &mut inferred_dim,
            )?;
        }
    }

    if rows.len() < needed_rows {
        return Err(format!(
            "dataset only yielded {} rows, but benchmark needs {}",
            rows.len(),
            needed_rows
        ));
    }

    let dim = inferred_dim.ok_or_else(|| {
        format!(
            "failed to infer dimension from dataset column `{column_name}` under {}",
            path.display()
        )
    })?;
    let dataset = rows[..vectors].to_vec();
    let queries = rows[query_offset..query_offset + queries].to_vec();
    Ok((dataset, queries, dim))
}

fn append_record_batches<E>(
    reader: impl IntoIterator<Item = Result<arrow_array::RecordBatch, E>>,
    source_kind: &str,
    file_path: &Path,
    column_name: &str,
    needed_rows: usize,
    rows: &mut Vec<Vec<f32>>,
    inferred_dim: &mut Option<usize>,
) -> Result<(), String>
where
    E: std::fmt::Display,
{
    for batch in reader {
        let batch = batch.map_err(|err| {
            format!(
                "failed to read {source_kind} {}: {err}",
                file_path.display()
            )
        })?;
        let column_index = batch.schema().index_of(column_name).map_err(|err| {
            format!(
                "column `{column_name}` not found in {}: {err}",
                file_path.display()
            )
        })?;
        let column = batch.column(column_index);
        if let Some(list) = column.as_any().downcast_ref::<GenericListArray<i32>>() {
            append_list_rows(list, needed_rows, rows, inferred_dim)?;
        } else if let Some(list) = column.as_any().downcast_ref::<GenericListArray<i64>>() {
            append_list_rows(list, needed_rows, rows, inferred_dim)?;
        } else {
            return Err(format!(
                "column `{column_name}` in {} is not a supported list array",
                file_path.display()
            ));
        }
        if rows.len() >= needed_rows {
            break;
        }
    }
    Ok(())
}

fn append_list_rows<OffsetSize: OffsetSizeTrait>(
    list: &GenericListArray<OffsetSize>,
    needed_rows: usize,
    rows: &mut Vec<Vec<f32>>,
    inferred_dim: &mut Option<usize>,
) -> Result<(), String> {
    let values = list.values();
    if let Some(values) = values.as_any().downcast_ref::<Float32Array>() {
        return append_typed_rows(
            list,
            values,
            needed_rows,
            rows,
            inferred_dim,
            |values, index| values.value(index),
        );
    }
    if let Some(values) = values.as_any().downcast_ref::<Float64Array>() {
        return append_typed_rows(
            list,
            values,
            needed_rows,
            rows,
            inferred_dim,
            |values, index| values.value(index) as f32,
        );
    }

    Err("only float32 and float64 embedding lists are supported".to_owned())
}

fn append_typed_rows<OffsetSize: OffsetSizeTrait, Values>(
    list: &GenericListArray<OffsetSize>,
    values: &Values,
    needed_rows: usize,
    rows: &mut Vec<Vec<f32>>,
    inferred_dim: &mut Option<usize>,
    read_value: impl Fn(&Values, usize) -> f32,
) -> Result<(), String> {
    for row_index in 0..list.len() {
        if rows.len() >= needed_rows {
            break;
        }
        rows.push(extract_row(
            list,
            values,
            row_index,
            inferred_dim,
            &read_value,
        )?);
    }

    Ok(())
}

fn collect_arrow_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    let metadata =
        fs::metadata(path).map_err(|err| format!("failed to stat {}: {err}", path.display()))?;
    if metadata.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    let mut files: Vec<_> = fs::read_dir(path)
        .map_err(|err| format!("failed to read directory {}: {err}", path.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|entry| entry.extension().is_some_and(|ext| ext == "arrow"))
        .collect();
    files.sort();

    if files.is_empty() {
        return Err(format!(
            "no `.arrow` files found under dataset path {}",
            path.display()
        ));
    }
    Ok(files)
}

fn extract_row<OffsetSize: OffsetSizeTrait, Values>(
    list: &GenericListArray<OffsetSize>,
    values: &Values,
    row_index: usize,
    inferred_dim: &mut Option<usize>,
    read_value: impl Fn(&Values, usize) -> f32,
) -> Result<Vec<f32>, String> {
    let offsets = list.value_offsets();
    let start = offsets[row_index].as_usize();
    let end = offsets[row_index + 1].as_usize();
    let row_dim = end - start;
    let dim = inferred_dim.get_or_insert(row_dim);
    if row_dim != *dim {
        return Err(format!(
            "row {row_index} has dim {}, expected {dim}",
            row_dim
        ));
    }
    Ok((start..end)
        .map(|index| read_value(values, index))
        .collect())
}
