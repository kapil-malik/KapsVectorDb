import argparse
import shutil
import time
from pathlib import Path
from statistics import mean

import numpy as np
from tqdm import tqdm

from vectordb.models import VectorRecord
from vectordb.store_base import VectorStore
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore
from vectordb.stores.file_backed import FileBackedVectorStore
from vectordb.stores.ivf_inmem import IVFVectorStore
from vectordb.stores.matrix_inmem import MatrixBackedInMemVectorStore
from vectordb.stores.mmap_store import MMapVectorStore
from vectordb.stores.naive_inmem import NaiveInMemVectorStore
from vectordb.stores.normalized_inmem import NormalizedInMemVectorStore


def clean_file_store_data(path: str = "benchmark_file_store") -> None:
    store_path = Path(path)

    if store_path.exists():
        shutil.rmtree(store_path)

    store_path.mkdir(parents=True, exist_ok=True)


def generate_random_vector(dim: int) -> np.ndarray:
    """
    Generate one random vector.

    We use float32 because real embedding vectors are commonly stored as float32,
    not Python float/double.
    """
    return np.random.random(dim).astype(np.float32)


def generate_records(num_records: int, dim: int) -> list[VectorRecord]:
    records = []

    for i in tqdm(range(num_records), desc="Generating vectors"):
        records.append(
            VectorRecord(
                id=f"record-{i}",
                vector=np.random.random(dim).astype(np.float32),
                text=f"Synthetic text for record {i}",
                metadata={"source": "synthetic"},
            )
        )

    return records


def percentile(values: list[float], p: float) -> float:
    """
    Simple percentile helper.

    Example:
    p=50 gives median.
    p=95 gives 95th percentile.
    """
    if not values:
        raise ValueError("values must not be empty")

    sorted_values = sorted(values)
    index = int((p / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def benchmark_insert(
        store: VectorStore,
        records: list[VectorRecord],
) -> None:
    start = time.perf_counter()

    for record in tqdm(records, desc="Inserting vectors"):
        store.insert(record)

    end = time.perf_counter()

    total_time_sec = end - start
    num_records = len(records)

    print("\nInsert benchmark")
    print("----------------")
    print(f"records       : {num_records}")
    print(f"total time    : {total_time_sec:.4f} sec")
    print(f"records/sec   : {num_records / total_time_sec:.2f}")


def benchmark_search(
        store: VectorStore,
        num_queries: int,
        dim: int,
        top_k: int,
) -> None:
    latencies_ms: list[float] = []

    # Warmup queries.
    # This avoids measuring some one-time overhead.
    for _ in range(5):
        query_vector = generate_random_vector(dim)
        store.search(query_vector=query_vector, top_k=top_k)

    for _ in tqdm(range(num_queries), desc="Searching vectors"):
        query_vector = generate_random_vector(dim)

        start = time.perf_counter()
        store.search(query_vector=query_vector, top_k=top_k)
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000)

    total_search_time_sec = sum(latencies_ms) / 1000
    queries_per_sec = num_queries / total_search_time_sec

    print("\nSearch benchmark")
    print("----------------")
    print(f"queries        : {num_queries}")
    print(f"top_k          : {top_k}")
    print(f"avg latency    : {mean(latencies_ms):.4f} ms")
    print(f"p50 latency    : {percentile(latencies_ms, 50):.4f} ms")
    print(f"p95 latency    : {percentile(latencies_ms, 95):.4f} ms")
    print(f"p99 latency    : {percentile(latencies_ms, 99):.4f} ms")
    print(f"queries/sec    : {queries_per_sec:.2f}")


def benchmark_delete(
        store: VectorStore,
        delete_count: int,
) -> None:
    start = time.perf_counter()

    deleted = 0

    for i in tqdm(range(delete_count), desc="Deleting vectors"):
        if store.delete(f"record-{i}"):
            deleted += 1

    end = time.perf_counter()

    total_time_sec = end - start

    print("\nDelete benchmark")
    print("----------------")
    print(f"requested deletes : {delete_count}")
    print(f"actual deletes    : {deleted}")
    print(f"total time        : {total_time_sec:.4f} sec")
    print(f"deletes/sec       : {deleted / total_time_sec:.2f}" if total_time_sec > 0 else "deletes/sec       : inf")
    print(f"remaining count   : {store.count()}")


def benchmark_compact(store: VectorStore) -> None:
    if not hasattr(store, "compact"):
        print("\nCompact benchmark")
        print("-----------------")
        print("Store does not support compact(); skipping.")
        return

    if hasattr(store, "save"):
        store.save()

    start = time.perf_counter()
    store.compact()
    end = time.perf_counter()

    total_time_sec = end - start

    print("\nCompact benchmark")
    print("-----------------")
    print(f"total time      : {total_time_sec:.4f} sec")
    print(f"remaining count : {store.count()}")


def benchmark_build(store: VectorStore) -> None:
    if not hasattr(store, "build"):
        print("\nBuild benchmark")
        print("----------------")
        print("Store does not support build(); skipping.")
        return

    if hasattr(store, "save"):
        store.save()

    start = time.perf_counter()
    store.build()
    end = time.perf_counter()

    total_time_sec = end - start

    print("\nBuild benchmark")
    print("----------------")
    print(f"total time      : {total_time_sec:.4f} sec")
    print(f"remaining count : {store.count()}")


def create_store(store_type: str) -> VectorStore:
    if store_type == "naive":
        return NaiveInMemVectorStore()
    elif store_type == "normalized":
        return NormalizedInMemVectorStore()
    elif store_type == "matrix":
        return MatrixBackedInMemVectorStore()
    elif store_type == "buffered-matrix":
        return BufferedMatrixInMemVectorStore()
    elif store_type == "ivf":
        return IVFVectorStore(nlist=100, nprobe=5, buffer_size=1024)
    elif store_type == "file":
        return FileBackedVectorStore(
            records_file="benchmark_file_store/records.jsonl",
            vectors_file="benchmark_file_store/vectors.npy",
            tombstones_file="benchmark_file_store/tombstones.txt",
        )
    elif store_type == "mmap":
        return MMapVectorStore(
            records_file="benchmark_file_store/records.jsonl",
            vectors_file="benchmark_file_store/vectors.npy",
            tombstones_file="benchmark_file_store/tombstones.txt",
        )

    raise ValueError(f"Unknown store type: {store_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vector store"
    )

    parser.add_argument("--store",
                        choices=["naive", "normalized", "matrix", "buffered-matrix", "ivf", "file", "mmap"],
                        default="naive")
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--delete-count", type=int, default=0)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--skip-insert", action="store_true")
    parser.add_argument("--clean-file-store", action="store_true")

    args = parser.parse_args()

    if args.store in ["file", "mmap"] and args.clean_file_store:
        clean_file_store_data()

    store = create_store(args.store)

    if not args.skip_insert:
        records = generate_records(args.records, args.dim)

        if args.delete_count > 0:
            undeleted_records = records[args.delete_count:]
            deleted_records = records[:args.delete_count]
        else:
            undeleted_records = records
            deleted_records = []

        benchmark_insert(store, undeleted_records)
        if hasattr(store, "save"):
            store.save()
        benchmark_build(store)

        if args.delete_count > 0:
            print(f"\nSearch on {store.count()} records. INSERT ONLY (no deletes yet)")
            benchmark_search(
                store=store,
                num_queries=args.queries,
                dim=args.dim,
                top_k=args.top_k,
            )

            benchmark_insert(store, deleted_records)
            if hasattr(store, "save"):
                store.save()
            benchmark_build(store)

    print(f"\nSearch on {store.count()} records. INSERT ONLY (no deletes yet)")
    benchmark_search(
        store=store,
        num_queries=args.queries,
        dim=args.dim,
        top_k=args.top_k,
    )

    if args.delete_count > 0:
        benchmark_delete(store, args.delete_count)

        print(f"\nSearch on {store.count()} records. AFTER DELETES")
        benchmark_search(
            store=store,
            num_queries=args.queries,
            dim=args.dim,
            top_k=args.top_k,
        )

    if args.compact:
        benchmark_compact(store)

        print(f"\nSearch on {store.count()} records. AFTER COMPACT")
        benchmark_search(
            store=store,
            num_queries=args.queries,
            dim=args.dim,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()