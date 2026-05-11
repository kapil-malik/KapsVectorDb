import argparse
import time
from statistics import mean

import numpy as np
from tqdm import tqdm

from vectordb.models import VectorRecord
from vectordb.store_base import VectorStore
from vectordb.stores.naive_inmem import NaiveInMemVectorStore
from vectordb.stores.normalized_inmem import NormalizedInMemVectorStore
from vectordb.stores.matrix_inmem import MatrixBackedInMemVectorStore
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore


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


def create_store(store_type: str) -> VectorStore:
    if store_type == "naive":
        return NaiveInMemVectorStore()
    elif store_type == "normalized":
        return NormalizedInMemVectorStore()
    elif store_type == "matrix":
        return MatrixBackedInMemVectorStore()
    elif store_type == "buffered-matrix":
        return BufferedMatrixInMemVectorStore()

    raise ValueError(f"Unknown store type: {store_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vector store"
    )

    parser.add_argument("--store",
                        choices=["naive", "normalized", "matrix", "buffered-matrix"],
                        default="naive")
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    records = generate_records(args.records, args.dim)
    store = create_store(args.store)

    benchmark_insert(store, records)

    benchmark_search(
        store=store,
        num_queries=args.queries,
        dim=args.dim,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()