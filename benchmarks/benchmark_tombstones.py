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

from benchmarks.benchmark_helpers import (
    generate_random_vector,
    generate_records,
    insert_records,
    percentile,
    search_with_latency,
)
from vectordb.stores.file_backed import FileBackedVectorStore


STORE_DATA_ROOT = Path("benchmark_store_data")


@dataclass(frozen=True)
class TombstoneStageResult:
    stage: str
    logical_records: int
    physical_rows: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    queries_per_sec: float


def _store_kwargs() -> dict:
    d = STORE_DATA_ROOT / "tombstones"
    return dict(
        records_file=str(d / "records.jsonl"),
        vectors_file=str(d / "vectors.npy"),
        tombstones_file=str(d / "tombstones.txt"),
    )


def _run_search_stage(
        store: FileBackedVectorStore,
        query_vectors: list[np.ndarray],
        top_k: int,
        stage: str,
) -> TombstoneStageResult:
    for qv in query_vectors[:5]:
        store.search(query_vector=qv, top_k=top_k)

    latencies: list[float] = []
    for qv in tqdm(query_vectors, desc="Searching"):
        _, ms = search_with_latency(store, qv, top_k)
        latencies.append(ms)

    total_sec = sum(latencies) / 1000
    avg = mean(latencies)
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)
    qps = len(latencies) / total_sec

    logical = store.count()
    physical = len(store._ids)

    print(f"\nSearch benchmark")
    print(f"  logical records : {logical:,}")
    print(f"  physical rows   : {physical:,}")
    print(f"  avg latency     : {avg:.4f} ms")
    print(f"  p50 latency     : {p50:.4f} ms")
    print(f"  p95 latency     : {p95:.4f} ms")
    print(f"  p99 latency     : {p99:.4f} ms")
    print(f"  queries/sec     : {qps:.0f}")

    return TombstoneStageResult(
        stage=stage,
        logical_records=logical,
        physical_rows=physical,
        avg_latency_ms=avg,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        queries_per_sec=qps,
    )


def _benchmark_delete(store: FileBackedVectorStore, ids: list[str]) -> None:
    start = time.perf_counter()
    deleted = 0
    for rid in tqdm(ids, desc="Deleting records"):
        if store.delete(rid):
            deleted += 1
    elapsed = time.perf_counter() - start

    rate = f"{deleted / elapsed:.0f}" if elapsed > 0 else "inf"
    print(f"\nDelete benchmark")
    print(f"  requested  : {len(ids):,}")
    print(f"  deleted    : {deleted:,}")
    print(f"  time       : {elapsed:.4f} sec")
    print(f"  rate       : {rate} deletes/sec")


def _benchmark_compact(store: FileBackedVectorStore) -> None:
    start = time.perf_counter()
    store.compact()
    elapsed = time.perf_counter() - start

    print(f"\nCompact benchmark")
    print(f"  time            : {elapsed:.4f} sec")
    print(f"  remaining count : {store.count():,}")


def _print_stage_table(stages: list[TombstoneStageResult]) -> None:
    print("\n\nTombstone & Compaction Stage Summary")
    print("=====================================")
    header = (
        f"{'Stage':<22} {'Logical':>9} {'Physical':>9} "
        f"{'Avg(ms)':>9} {'p50(ms)':>9} {'p95(ms)':>9} {'p99(ms)':>9} {'QPS':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in stages:
        print(
            f"{r.stage:<22} "
            f"{r.logical_records:>9,} "
            f"{r.physical_rows:>9,} "
            f"{r.avg_latency_ms:>9.4f} "
            f"{r.p50_latency_ms:>9.4f} "
            f"{r.p95_latency_ms:>9.4f} "
            f"{r.p99_latency_ms:>9.4f} "
            f"{r.queries_per_sec:>7.0f}"
        )


def write_stages_csv(output_csv: str, stages: list[TombstoneStageResult]) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_dicts = [dataclasses.asdict(r) for r in stages]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)

    print(f"\nWrote tombstone stage summary to: {output_path}")


def run_tombstone_benchmark(
        args,
        records: list,
        query_vectors: list[np.ndarray],
) -> list[TombstoneStageResult]:
    # survivors: inserted first, never deleted
    # doomed: inserted second in stage 2, deleted in stage 3
    survivors = records[args.delete_count:]
    doomed = records[:args.delete_count]

    data_dir = STORE_DATA_ROOT / "tombstones"
    if args.clean:
        if data_dir.exists():
            shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    store = FileBackedVectorStore(**_store_kwargs())
    stages: list[TombstoneStageResult] = []

    # Stage 1: insert survivors, search
    print(f"\n{'='*50}")
    print(f"Stage 1: Insert {len(survivors):,} survivor records")
    print(f"{'='*50}")
    elapsed = insert_records(store, survivors, desc="Inserting survivor records")
    print(f"  {len(survivors):,} records in {elapsed:.2f}s ({len(survivors)/elapsed:,.0f} rec/s)")
    store.save()
    stages.append(_run_search_stage(store, query_vectors, args.top_k, "after_first_insert"))

    # Stage 2: insert doomed records too, search
    print(f"\n{'='*50}")
    print(f"Stage 2: Insert {len(doomed):,} more records (total {args.records:,})")
    print(f"{'='*50}")
    elapsed = insert_records(store, doomed, desc="Inserting doomed records")
    print(f"  {len(doomed):,} records in {elapsed:.2f}s ({len(doomed)/elapsed:,.0f} rec/s)")
    store.save()
    stages.append(_run_search_stage(store, query_vectors, args.top_k, "after_full_insert"))

    # Stage 3: delete doomed records, search
    print(f"\n{'='*50}")
    print(f"Stage 3: Delete {len(doomed):,} records")
    print(f"{'='*50}")
    _benchmark_delete(store, [r.id for r in doomed])
    stages.append(_run_search_stage(store, query_vectors, args.top_k, "after_delete"))

    # Stage 4: compact, search
    print(f"\n{'='*50}")
    print("Stage 4: Compact")
    print(f"{'='*50}")
    _benchmark_compact(store)
    stages.append(_run_search_stage(store, query_vectors, args.top_k, "after_compact"))

    return stages


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark how tombstones and compaction affect FileBackedVectorStore search latency. "
            "Runs four sequential stages: initial insert → full insert → delete → compact. "
            "Tracks both logical record count and physical matrix rows at each stage."
        )
    )
    parser.add_argument(
        "--records",
        type=int,
        default=100_000,
        help="Total vectors to insert across both insert stages (default: 100,000).",
    )
    parser.add_argument(
        "--delete-count",
        type=int,
        default=60_000,
        help=(
            "Records inserted in stage 2 then deleted in stage 3 (default: 60,000). "
            "Must be less than --records."
        ),
    )
    parser.add_argument("--dim", type=int, default=384, help="Vector dimensions (default: 384).")
    parser.add_argument("--queries", type=int, default=1_000, help="Search queries per stage (default: 1,000).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe benchmark_store_data/tombstones/ before running.",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmarks/results/tombstones_compaction.csv",
        help="Path to write the stage summary CSV (default: benchmarks/results/tombstones_compaction.csv).",
    )

    args = parser.parse_args()

    if args.delete_count >= args.records:
        parser.error("--delete-count must be less than --records")
    if args.delete_count <= 0:
        parser.error("--delete-count must be positive")

    records = generate_records(args.records, args.dim, desc="Generating vectors")
    query_vectors = [generate_random_vector(args.dim) for _ in range(args.queries)]

    stages = run_tombstone_benchmark(args, records, query_vectors)

    _print_stage_table(stages)
    write_stages_csv(args.output_csv, stages)


if __name__ == "__main__":
    main()
