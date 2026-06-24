import argparse
import csv
import dataclasses
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from tqdm import tqdm

from benchmarks.benchmark_helpers import generate_random_vector, generate_records, percentile

from vectordb.models import VectorRecord
from vectordb.store_base import VectorStore
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore
from vectordb.stores.file_backed import FileBackedVectorStore
from vectordb.stores.matrix_inmem import MatrixBackedInMemVectorStore
from vectordb.stores.mmap_store import MMapVectorStore
from vectordb.stores.naive_inmem import NaiveInMemVectorStore
from vectordb.stores.normalized_inmem import NormalizedInMemVectorStore


EXACT_STORES = ["naive", "normalized", "matrix", "buffered-matrix", "file", "mmap"]

STORE_DATA_ROOT = Path("benchmark_store_data")


@dataclass(frozen=True)
class StoreBenchmarkResult:
    store: str
    records: int
    dim: int
    queries: int
    top_k: int
    insert_time_sec: float
    insert_throughput_rec_per_sec: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    queries_per_sec: float


def store_data_dir(store_type: str) -> Path:
    return STORE_DATA_ROOT / store_type


def clean_store_data(store_type: str) -> None:
    path = store_data_dir(store_type)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _file_store_kwargs(store_type: str) -> dict:
    d = store_data_dir(store_type)
    return dict(
        records_file=str(d / "records.jsonl"),
        vectors_file=str(d / "vectors.npy"),
        tombstones_file=str(d / "tombstones.txt"),
    )


def create_store(store_type: str) -> VectorStore:
    if store_type == "naive":
        return NaiveInMemVectorStore()
    if store_type == "normalized":
        return NormalizedInMemVectorStore()
    if store_type == "matrix":
        return MatrixBackedInMemVectorStore()
    if store_type == "buffered-matrix":
        return BufferedMatrixInMemVectorStore()
    if store_type == "file":
        return FileBackedVectorStore(**_file_store_kwargs("file"))
    if store_type == "mmap":
        return MMapVectorStore(**_file_store_kwargs("mmap"))
    raise ValueError(f"Unknown store type: {store_type}")


def benchmark_insert(
        store: VectorStore,
        records: list[VectorRecord],
) -> tuple[float, float]:
    """Returns (total_time_sec, throughput_rec_per_sec)."""
    start = time.perf_counter()

    for record in tqdm(records, desc="Inserting vectors"):
        store.insert(record)

    end = time.perf_counter()

    total_time_sec = end - start
    num_records = len(records)
    throughput = num_records / total_time_sec

    print("\nInsert benchmark")
    print("----------------")
    print(f"records       : {num_records}")
    print(f"total time    : {total_time_sec:.4f} sec")
    print(f"records/sec   : {throughput:.2f}")

    return total_time_sec, throughput


def benchmark_search(
        store: VectorStore,
        query_vectors: list[np.ndarray],
        top_k: int,
) -> tuple[float, float, float, float, float]:
    """Returns (avg_ms, p50_ms, p95_ms, p99_ms, queries_per_sec)."""
    latencies_ms: list[float] = []

    for qv in query_vectors[:5]:
        store.search(query_vector=qv, top_k=top_k)

    for qv in tqdm(query_vectors, desc="Searching vectors"):
        start = time.perf_counter()
        store.search(query_vector=qv, top_k=top_k)
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000)

    num_queries = len(query_vectors)
    total_search_time_sec = sum(latencies_ms) / 1000
    queries_per_sec = num_queries / total_search_time_sec
    avg_ms = mean(latencies_ms)
    p50_ms = percentile(latencies_ms, 50)
    p95_ms = percentile(latencies_ms, 95)
    p99_ms = percentile(latencies_ms, 99)

    print("\nSearch benchmark")
    print("----------------")
    print(f"queries        : {num_queries}")
    print(f"top_k          : {top_k}")
    print(f"avg latency    : {avg_ms:.4f} ms")
    print(f"p50 latency    : {p50_ms:.4f} ms")
    print(f"p95 latency    : {p95_ms:.4f} ms")
    print(f"p99 latency    : {p99_ms:.4f} ms")
    print(f"queries/sec    : {queries_per_sec:.2f}")

    return avg_ms, p50_ms, p95_ms, p99_ms, queries_per_sec


def benchmark_build(store: VectorStore) -> None:
    if not hasattr(store, "build"):
        print("\nBuild benchmark")
        print("---------------")
        print("Store does not support build(); skipping.")
        return

    if hasattr(store, "save"):
        store.save()

    start = time.perf_counter()
    store.build()
    end = time.perf_counter()

    print("\nBuild benchmark")
    print("---------------")
    print(f"total time      : {end - start:.4f} sec")
    print(f"remaining count : {store.count()}")


def run_store_benchmark(
        store_type: str,
        args,
        records: list[VectorRecord],
        query_vectors: list[np.ndarray],
) -> StoreBenchmarkResult:
    print(f"\n{'=' * 50}")
    print(f"Store: {store_type}")
    print(f"{'=' * 50}")

    if store_type in ("file", "mmap") and args.clean_file_store:
        clean_store_data(store_type)
    elif store_type in ("file", "mmap"):
        store_data_dir(store_type).mkdir(parents=True, exist_ok=True)

    # MMapVectorStore is read-only: inserts go through a FileBackedVectorStore
    # using the same data directory, then the mmap store reloads for search.
    if store_type == "mmap":
        insert_store = FileBackedVectorStore(**_file_store_kwargs("mmap"))
    else:
        insert_store = create_store(store_type)

    insert_time_sec, insert_throughput = benchmark_insert(insert_store, records)
    if hasattr(insert_store, "save"):
        insert_store.save()
    benchmark_build(insert_store)

    # For mmap: reload the saved files via MMapVectorStore for the search phase.
    if store_type == "mmap":
        print("\nReloading data as MMapVectorStore for search benchmark...")
        store = MMapVectorStore(**_file_store_kwargs("mmap"))
    else:
        store = insert_store

    print(f"\nSearch on {store.count()} records.")
    avg_ms, p50_ms, p95_ms, p99_ms, qps = benchmark_search(
        store=store, query_vectors=query_vectors, top_k=args.top_k,
    )

    return StoreBenchmarkResult(
        store=store_type,
        records=args.records,
        dim=args.dim,
        queries=args.queries,
        top_k=args.top_k,
        insert_time_sec=insert_time_sec,
        insert_throughput_rec_per_sec=insert_throughput,
        avg_latency_ms=avg_ms,
        p50_latency_ms=p50_ms,
        p95_latency_ms=p95_ms,
        p99_latency_ms=p99_ms,
        queries_per_sec=qps,
    )


def write_summary_csv(output_csv: str, rows: list[StoreBenchmarkResult]) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_dicts = [dataclasses.asdict(r) for r in rows]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)

    print(f"\nWrote store comparison summary to: {output_path}")


def print_comparison_table(rows: list[StoreBenchmarkResult]) -> None:
    print("\n\nStore Comparison Summary")
    print("========================")
    header = f"{'Store':<18} {'Insert(s)':>10} {'Rec/s':>10} {'Avg(ms)':>9} {'p50(ms)':>9} {'p95(ms)':>9} {'p99(ms)':>9} {'QPS':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.store:<18} "
            f"{r.insert_time_sec:>10.4f} "
            f"{r.insert_throughput_rec_per_sec:>10.0f} "
            f"{r.avg_latency_ms:>9.4f} "
            f"{r.p50_latency_ms:>9.4f} "
            f"{r.p95_latency_ms:>9.4f} "
            f"{r.p99_latency_ms:>9.4f} "
            f"{r.queries_per_sec:>9.0f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark vector store")

    parser.add_argument(
        "--store",
        choices=EXACT_STORES + ["all"],
        default="naive",
        help=(
            "Store to benchmark. "
            "'all' runs all exact stores: " + ", ".join(EXACT_STORES)
        ),
    )
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--clean-file-store",
        action="store_true",
        help="Wipe and recreate data directories for file and mmap stores before running.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Path to write the summary CSV. "
            "Defaults to benchmarks/results/store_comparison.csv when --store all."
        ),
    )

    args = parser.parse_args()

    records = generate_records(args.records, args.dim, desc="Generating vectors")
    query_vectors = [generate_random_vector(args.dim) for _ in range(args.queries)]

    stores_to_run = EXACT_STORES if args.store == "all" else [args.store]

    results: list[StoreBenchmarkResult] = []
    for store_type in stores_to_run:
        results.append(run_store_benchmark(store_type, args, records, query_vectors))

    if args.store == "all":
        print_comparison_table(results)
        output_csv = args.output_csv or "benchmarks/results/store_comparison.csv"
        write_summary_csv(output_csv, results)


if __name__ == "__main__":
    main()
