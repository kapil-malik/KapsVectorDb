import argparse
import time
from statistics import mean

import numpy as np
from tqdm import tqdm

from vectordb.models import VectorRecord
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore
from vectordb.stores.ivf_inmem import IVFVectorStore


def generate_random_vector(dim: int) -> np.ndarray:
    return np.random.random(dim).astype(np.float32)


def generate_records(num_records: int, dim: int) -> list[VectorRecord]:
    records = []

    for i in tqdm(range(num_records), desc="Generating records"):
        records.append(
            VectorRecord(
                id=f"record-{i}",
                vector=generate_random_vector(dim),
                text=f"Synthetic text for record {i}",
                metadata={"source": "synthetic"},
            )
        )

    return records


def generate_queries(num_queries: int, dim: int) -> list[np.ndarray]:
    return [
        generate_random_vector(dim)
        for _ in tqdm(range(num_queries), desc="Generating queries")
    ]


def percentile(values: list[float], p: float) -> float:
    sorted_values = sorted(values)
    index = int((p / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def insert_records(store, records: list[VectorRecord]) -> None:
    for record in tqdm(records, desc=f"Inserting into {store.__class__.__name__}"):
        store.insert(record)


def search_with_latency(store, query: np.ndarray, top_k: int):
    start = time.perf_counter()
    results = store.search(query_vector=query, top_k=top_k)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    return results, latency_ms


def recall_at_k(exact_ids: list[str], candidate_ids: list[str], k: int) -> float:
    exact_top_k = set(exact_ids[:k])
    candidate_top_k = set(candidate_ids[:k])

    if not exact_top_k:
        return 0.0

    return len(exact_top_k.intersection(candidate_top_k)) / k


def benchmark_ivf_recall(
        records: list[VectorRecord],
        queries: list[np.ndarray],
        top_k: int,
        nlist: int,
        nprobe: int,
) -> None:
    exact_store = BufferedMatrixInMemVectorStore(buffer_size=1024)
    ivf_store = IVFVectorStore(
        nlist=nlist,
        nprobe=nprobe,
        buffer_size=1024,
    )

    insert_records(exact_store, records)
    insert_records(ivf_store, records)

    print("\nBuilding IVF index")
    print("------------------")
    start = time.perf_counter()
    ivf_store.build()
    end = time.perf_counter()
    print(f"build time : {end - start:.4f} sec")

    exact_latencies_ms: list[float] = []
    ivf_latencies_ms: list[float] = []
    recalls: list[float] = []

    # Warmup
    for query in queries[:5]:
        exact_store.search(query_vector=query, top_k=top_k)
        ivf_store.search(query_vector=query, top_k=top_k)

    for query in tqdm(queries, desc="Comparing exact vs IVF"):
        exact_results, exact_latency = search_with_latency(
            exact_store,
            query,
            top_k,
        )

        ivf_results, ivf_latency = search_with_latency(
            ivf_store,
            query,
            top_k,
        )

        exact_ids = [result.record.id for result in exact_results]
        ivf_ids = [result.record.id for result in ivf_results]

        exact_latencies_ms.append(exact_latency)
        ivf_latencies_ms.append(ivf_latency)
        recalls.append(recall_at_k(exact_ids, ivf_ids, top_k))

    print("\nIVF latency vs recall benchmark")
    print("-------------------------------")
    print(f"records       : {len(records)}")
    print(f"queries       : {len(queries)}")
    print(f"top_k         : {top_k}")
    print(f"nlist         : {nlist}")
    print(f"nprobe        : {nprobe}")

    print("\nExact baseline latency")
    print("----------------------")
    print(f"avg latency   : {mean(exact_latencies_ms):.4f} ms")
    print(f"p50 latency   : {percentile(exact_latencies_ms, 50):.4f} ms")
    print(f"p95 latency   : {percentile(exact_latencies_ms, 95):.4f} ms")
    print(f"p99 latency   : {percentile(exact_latencies_ms, 99):.4f} ms")

    print("\nIVF latency")
    print("-----------")
    print(f"avg latency   : {mean(ivf_latencies_ms):.4f} ms")
    print(f"p50 latency   : {percentile(ivf_latencies_ms, 50):.4f} ms")
    print(f"p95 latency   : {percentile(ivf_latencies_ms, 95):.4f} ms")
    print(f"p99 latency   : {percentile(ivf_latencies_ms, 99):.4f} ms")

    print("\nRecall")
    print("------")
    print(f"avg recall@{top_k} : {mean(recalls):.4f}")
    print(f"p50 recall@{top_k} : {percentile(recalls, 50):.4f}")
    print(f"p95 recall@{top_k} : {percentile(recalls, 95):.4f}")

    speedup = mean(exact_latencies_ms) / mean(ivf_latencies_ms)
    print("\nSummary")
    print("-------")
    print(f"speedup       : {speedup:.2f}x")
    print(f"recall@{top_k}     : {mean(recalls):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark IVF latency vs exact-search recall"
    )

    parser.add_argument("--records", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=5)

    args = parser.parse_args()

    records = generate_records(args.records, args.dim)
    queries = generate_queries(args.queries, args.dim)

    benchmark_ivf_recall(
        records=records,
        queries=queries,
        top_k=args.top_k,
        nlist=args.nlist,
        nprobe=args.nprobe,
    )


if __name__ == "__main__":
    main()