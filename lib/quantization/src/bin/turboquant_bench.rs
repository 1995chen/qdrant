//! Standalone TurboQuant benchmark CLI.
//!
//! The goal of this binary is to make algorithm iteration fast while the
//! implementation is still isolated from the rest of Qdrant. We deliberately
//! avoid external CLI dependencies here so the tool stays lightweight.
//!
//! Example:
//! `cargo run -p quantization --bin turboquant_bench -- --dataset-path ./dataset --vectors 2000 --queries 200 --bits 3`

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use arrow_array::{Array, Float32Array, Float64Array, GenericListArray, OffsetSizeTrait};
use arrow_ipc::reader::{FileReader, StreamReader};
use quantization::EncodingError;
use quantization::turboquant::{
    NormCorrection, TurboQuantCodec, TurboQuantConfig, TurboQuantVector, simd,
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

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn print_variant_row<T>(name: &str, report: &T, dim: usize, qjl_bits: usize)
where
    T: IReport,
{
    let recall_10 = report.recall(10).unwrap_or(0.0);
    let recall_100 = report.recall(100).unwrap_or(0.0);
    let encode_time = report
        .encode_time()
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "-".to_owned());
    let decode_time = report
        .decode_time()
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "-".to_owned());
    let search_time = report
        .search_time()
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "-".to_owned());
    println!(
        "{:<24} {:>10.4} {:>10.4} {:>11} {:>11} {:>11} {:>12} {:>12}",
        name, recall_10, recall_100, encode_time, decode_time, search_time, dim, qjl_bits
    );
}

fn build_variants(dim: usize, bits: u8, seed: u64) -> Vec<Variant> {
    let scalar = Variant {
        name: "default/Haar".into(),
        config: TurboQuantConfig::new(dim, bits, seed),
        use_simd: false,
    };

    let qjl_dense = Variant {
        name: "qjl/Haar".into(),
        config: TurboQuantConfig::new(dim, bits, seed).with_qjl(true),
        use_simd: false,
    };

    let qjl_norm_dense = Variant {
        name: "qjl+norm/Haar".into(),
        config: TurboQuantConfig::new(dim, bits, seed)
            .with_qjl(true)
            .with_norm_correction(NormCorrection::Exact),
        use_simd: false,
    };

    let norm = Variant {
        name: "norm/Haar".into(),
        config: TurboQuantConfig::new(dim, bits, seed).with_norm_correction(NormCorrection::Exact),
        use_simd: false,
    };

    let simd = Variant {
        name: "simd/Haar".into(),
        config: TurboQuantConfig::new(dim, bits, seed).with_norm_correction(NormCorrection::Exact),
        use_simd: true,
    };

    vec![scalar, qjl_dense, qjl_norm_dense, norm, simd]
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

trait IReport {
    fn recall(&self, k: usize) -> Option<f32>;
    fn encode_time(&self) -> Option<f64>;
    fn decode_time(&self) -> Option<f64>;
    fn search_time(&self) -> Option<f64>;
}

struct BaseLineReport {
    pub recall_data: Vec<Vec<usize>>,
    pub search_time: Option<f64>,
}
impl IReport for BaseLineReport {
    fn recall(&self, _: usize) -> Option<f32> {
        return Some(1.0);
    }
    fn encode_time(&self) -> Option<f64> {
        return None;
    }
    fn decode_time(&self) -> Option<f64> {
        return None;
    }
    fn search_time(&self) -> Option<f64> {
        return self.search_time;
    }
}

struct ReportItem {
    pub k: usize,
    pub recall: f32,
}

struct TurboQuantReport {
    pub all: Vec<ReportItem>,
    pub encode_time: Option<f64>,
    pub decode_time: Option<f64>,
    pub search_time: Option<f64>,
}

impl TurboQuantReport {
    #[must_use]
    pub fn with_encode_time(mut self, encode_time: Option<f64>) -> Self {
        self.encode_time = encode_time;
        self
    }
}

impl IReport for TurboQuantReport {
    fn recall(&self, k: usize) -> Option<f32> {
        self.all
            .iter()
            .find(|entry| entry.k == k)
            .map(|entry| entry.recall)
    }
    fn encode_time(&self) -> Option<f64> {
        return self.encode_time;
    }
    fn decode_time(&self) -> Option<f64> {
        return self.decode_time;
    }
    fn search_time(&self) -> Option<f64> {
        return self.search_time;
    }
}

fn recall_for_baseline<F>(
    original: &[Vec<f32>],
    queries: &[Vec<f32>],
    ks: &[usize],
    score_fn: F,
) -> BaseLineReport
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    let max_k = ks.iter().copied().max().unwrap_or(0);
    let mut recall_data = Vec::with_capacity(queries.len());
    let mut search_time: f64 = 0.0;

    for query in queries {
        let started = Instant::now();
        let mut exact_scores: Vec<(usize, f32)> = original
            .iter()
            .enumerate()
            .map(|(index, vector)| (index, score_fn(query, vector)))
            .collect();
        search_time += duration_ms(started.elapsed());
        exact_scores.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
        recall_data.push(
            exact_scores
                .iter()
                .take(max_k)
                .map(|&(index, _)| index)
                .collect(),
        );
    }
    BaseLineReport {
        recall_data,
        search_time: Some(search_time),
    }
}

fn recall_for_turboquant<F>(
    codec: &TurboQuantCodec,
    encoded: &[TurboQuantVector],
    queries: &[Vec<f32>],
    ks: &[usize],
    score_fn: F,
    baseline_result: &BaseLineReport,
) -> Result<TurboQuantReport, EncodingError>
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    let mut hit_counts: BTreeMap<usize, usize> = ks.iter().copied().map(|k| (k, 0)).collect();
    let total = queries.len();
    let dim = codec.config().dim();

    let started = Instant::now();
    let mut dequantized_vectors = vec![0.0f32; encoded.len() * dim];
    for (vector, output) in encoded
        .iter()
        .zip(dequantized_vectors.chunks_exact_mut(dim))
    {
        codec.dequantize_into(vector, output)?;
    }

    let decode_time: f64 = duration_ms(started.elapsed());
    let mut search_time: f64 = 0.0;
    for (query, exact_top_k) in queries.iter().zip(&baseline_result.recall_data) {
        let started = Instant::now();
        let mut approx_scores: Vec<(usize, f32)> = dequantized_vectors
            .chunks_exact(dim)
            .enumerate()
            .map(|(index, vector)| (index, score_fn(query, vector)))
            .collect();
        search_time += duration_ms(started.elapsed());
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

    let all_report = ks
        .iter()
        .map(|&k| ReportItem {
            k: k,
            recall: hit_counts[&k] as f32 / (total * k) as f32,
        })
        .collect();

    Ok(TurboQuantReport {
        all: all_report,
        encode_time: None,
        decode_time: Some(decode_time),
        search_time: Some(search_time),
    })
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
    let baseline_report =
        recall_for_baseline(&dataset, &queries, &ks, |v1, v2| simd::dot_plain(v1, v2));

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
        "decode_ms",
        "search_ms",
        "dim",
        "qjl_bits"
    );
    println!("{}", "-".repeat(108));
    print_variant_row("exact/f32", &baseline_report, dim, 0);

    for variant in variants {
        let codec = TurboQuantCodec::new(variant.config.clone()).map_err(|err| err.to_string())?;
        let encode_started = Instant::now();
        let encoded = codec
            .quantize_batch(&dataset)
            .map_err(|err| err.to_string())?;
        let encode_elapsed = encode_started.elapsed();
        let report = recall_for_turboquant(
            &codec,
            &encoded,
            &queries,
            &ks,
            |v1, v2| {
                if variant.use_simd {
                    return simd::dot(v1, v2);
                } else {
                    return simd::dot_plain(v1, v2);
                }
            },
            &baseline_report,
        )
        .map_err(|err| err.to_string())?;
        let report = report.with_encode_time(Some(duration_ms(encode_elapsed)));
        print_variant_row(
            &variant.name,
            &report,
            dim,
            if variant.config.qjl() { dim } else { 0 },
        );
    }

    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("turboquant_bench error: {error}");
        std::process::exit(1);
    }
}
