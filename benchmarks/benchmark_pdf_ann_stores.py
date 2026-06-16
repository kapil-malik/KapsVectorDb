import argparse
import csv
import itertools
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from vectordb.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from vectordb.ingestion.chunker import RecursiveTextChunker
from vectordb.ingestion.pdf_ingestion import chunks_from_pdf
from vectordb.models import SearchDiagnostics, VectorRecord
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore
from vectordb.stores.flat_nsw_inmem import FlatNSWVectorStore
from vectordb.stores.ivf_inmem import IVFVectorStore
from vectordb.stores.hnsw_inmem import HNSWVectorStore

@dataclass(frozen=True)
class BenchmarkRow:
    store: str
    store_parameters: str
    records: int
    queries: int
    insert_time_sec: float
    build_time_sec: float
    prepare_time_sec: float
    search_avg_latency_ms: float
    search_p50_latency_ms: float
    search_p95_latency_ms: float
    search_p99_latency_ms: float
    recall_avg: float
    recall_p50: float
    recall_p95: float
    diag_avg_distance_computations: int = 0
    diag_avg_visited_nodes: int = 0
    diag_avg_graph_hops: int = 0
    diag_avg_layers_traversed: int = 0
    diag_avg_clusters_scanned: int = 0
    diag_avg_vectors_scanned: int = 0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int((p / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def load_queries(path: str) -> list[str]:
    query_path = Path(path)

    if not query_path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    queries = [
        line.strip()
        for line in query_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not queries:
        raise ValueError(f"No queries found in {path}")

    return queries


def embed_texts(embedding_model, texts: list[str], desc: str):
    vectors = []

    for text in tqdm(texts, desc=desc):
        vectors.append(embedding_model.embed(text))

    return vectors


def build_records(chunks, vectors) -> list[VectorRecord]:
    records: list[VectorRecord] = []

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        metadata = dict(chunk.metadata)
        metadata["chunk_index"] = i

        records.append(
            VectorRecord(
                id=f"pdf-chunk-{i}",
                vector=vector,
                text=chunk.text,
                metadata=metadata,
            )
        )

    return records


def insert_records(store, records: list[VectorRecord]) -> float:
    start = time.perf_counter()

    for record in records:
        store.insert(record)

    end = time.perf_counter()
    return end - start


def maybe_build_store(store) -> float:
    if hasattr(store, "build"):
        start = time.perf_counter()
        store.build()
        end = time.perf_counter()
        return end - start

    return 0.0


def recall_at_k(exact_ids: list[str], candidate_ids: list[str], k: int) -> float:
    exact_top_k = set(exact_ids[:k])
    candidate_top_k = set(candidate_ids[:k])

    if not exact_top_k:
        return 0.0

    return len(exact_top_k.intersection(candidate_top_k)) / k


def avg_diagnostics(diags: list[SearchDiagnostics]) -> SearchDiagnostics:
    if not diags:
        return SearchDiagnostics()
    return SearchDiagnostics(
        distance_computations=round(mean(d.distance_computations for d in diags)),
        visited_nodes=round(mean(d.visited_nodes for d in diags)),
        graph_hops=round(mean(d.graph_hops for d in diags)),
        layers_traversed=round(mean(d.layers_traversed for d in diags)),
        clusters_scanned=round(mean(d.clusters_scanned for d in diags)),
        vectors_scanned=round(mean(d.vectors_scanned for d in diags)),
    )


def evaluate_store(
        store,
        query_vectors,
        exact_result_ids_by_query: list[list[str]] | None,
        top_k: int,
) -> tuple[list[float], list[float], SearchDiagnostics]:
    latencies_ms: list[float] = []
    recalls: list[float] = []
    diags: list[SearchDiagnostics] = []

    has_diagnostics = hasattr(store, "search_with_diagnostics")

    for i, query_vector in enumerate(query_vectors):
        start = time.perf_counter()
        if has_diagnostics:
            results, diag = store.search_with_diagnostics(query_vector=query_vector, top_k=top_k)
            diags.append(diag)
        else:
            results = store.search(query_vector=query_vector, top_k=top_k)
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000)

        if exact_result_ids_by_query is not None:
            candidate_ids = [result.record.id for result in results]
            recalls.append(
                recall_at_k(
                    exact_result_ids_by_query[i],
                    candidate_ids,
                    top_k,
                )
            )

    return latencies_ms, recalls, avg_diagnostics(diags)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def params_to_json(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark exact, IVF, Flat NSW, and HNSW stores on a real PDF corpus"
    )

    parser.add_argument("--pdf", required=True)
    parser.add_argument("--queries-file", required=True)
    parser.add_argument("--output-csv", default="pdf_ann_benchmark_results.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--buffer-size", type=int, default=1024)

    parser.add_argument("--ivf-nlist", default="10,50,100")
    parser.add_argument("--ivf-nprobe", default="1,5,10")

    parser.add_argument("--nsw-m", default="8,16")
    parser.add_argument("--nsw-ef-search", default="16,32,64")

    parser.add_argument("--hnsw-m", default="8,16")
    parser.add_argument("--hnsw-ef-construction", default="32,64,128")
    parser.add_argument("--hnsw-ef-search", default="16,32,64")
    parser.add_argument("--hnsw-level-multiplier", default="1.0")

    args = parser.parse_args()

    ivf_nlists = parse_int_list(args.ivf_nlist)
    ivf_nprobes = parse_int_list(args.ivf_nprobe)
    nsw_ms = parse_int_list(args.nsw_m)
    nsw_ef_search_values = parse_int_list(args.nsw_ef_search)
    hnsw_ms = parse_int_list(args.hnsw_m)
    hnsw_ef_construction_values = parse_int_list(args.hnsw_ef_construction)
    hnsw_ef_search_values = parse_int_list(args.hnsw_ef_search)
    hnsw_level_multiplier = float(args.hnsw_level_multiplier)

    buffer_size = args.buffer_size
    top_k = args.top_k

    embedding_model = SentenceTransformerEmbeddingModel()

    print("\nLoading PDF chunks")
    print("------------------")
    chunks = chunks_from_pdf(args.pdf, RecursiveTextChunker())
    chunk_texts = [chunk.text for chunk in chunks]

    print(f"PDF chunks: {len(chunks)}")

    queries = load_queries(args.queries_file)
    print(f"Queries   : {len(queries)}")

    print("\nEmbedding PDF chunks")
    print("--------------------")
    chunk_vectors = embed_texts(
        embedding_model,
        chunk_texts,
        "Embedding chunks",
    )

    print("\nEmbedding queries")
    print("-----------------")
    query_vectors = embed_texts(
        embedding_model,
        queries,
        "Embedding queries",
    )

    records = build_records(chunks, chunk_vectors)

    rows: list[dict[str, Any]] = []

    exact_benchmark_row, exact_result_ids_by_query = run_exact_benchmark(
        records=records,
        query_vectors=query_vectors,
        buffer_size=buffer_size,
        top_k=top_k
    )
    rows.append(exact_benchmark_row.__dict__)

    for nlist, nprobe in itertools.product(ivf_nlists, ivf_nprobes):
        if nprobe > nlist:
            continue

        benchmark_row = run_ivf_benchmark(
            records=records,
            query_vectors=query_vectors,
            exact_result_ids_by_query=exact_result_ids_by_query,
            buffer_size=buffer_size,
            top_k=top_k,
            nlist=nlist,
            nprobe=nprobe,
        )
        rows.append(benchmark_row.__dict__)


    for m, ef_search in itertools.product(nsw_ms, nsw_ef_search_values):
        benchmark_row = run_flat_nsw_benchmark(
            records=records,
            query_vectors=query_vectors,
            exact_result_ids_by_query=exact_result_ids_by_query,
            top_k=top_k,
            m=m,
            ef_search=ef_search,
        )
        rows.append(benchmark_row.__dict__)


    for m, ef_construction, ef_search in itertools.product(
        hnsw_ms,
        hnsw_ef_construction_values,
        hnsw_ef_search_values,
    ):
        benchmark_row = run_hnsw_benchmark(
            records=records,
            query_vectors=query_vectors,
            exact_result_ids_by_query=exact_result_ids_by_query,
            top_k=top_k,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            level_multiplier=hnsw_level_multiplier,
        )
        rows.append(benchmark_row.__dict__)

    output_path = Path(args.output_csv)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote benchmark results to: {output_path}")


def run_exact_benchmark(
        records: list[VectorRecord],
        query_vectors: list[np.ndarray],
        buffer_size: int,
        top_k: int) -> tuple[BenchmarkRow, list[list[str]]]:

    print("\nPreparing exact baseline")
    print("------------------------")
    exact_store = BufferedMatrixInMemVectorStore(buffer_size=buffer_size)
    exact_insert_time_sec = insert_records(exact_store, records)
    exact_build_time_sec = maybe_build_store(exact_store)
    # Warmup
    for query_vector in query_vectors[: min(5, len(query_vectors))]:
        exact_store.search(query_vector=query_vector, top_k=top_k)
    exact_latencies_ms, _, _ = evaluate_store(
        store=exact_store,
        query_vectors=query_vectors,
        exact_result_ids_by_query=None,
        top_k=top_k,
    )
    exact_result_ids_by_query = []
    for query_vector in query_vectors:
        exact_results = exact_store.search(query_vector=query_vector, top_k=top_k)
        exact_result_ids_by_query.append(
            [result.record.id for result in exact_results]
        )
    exact_benchmark_row = BenchmarkRow(
        store="exact",
        store_parameters=params_to_json({}),
        records=len(records),
        queries=len(query_vectors),
        insert_time_sec=exact_insert_time_sec,
        build_time_sec=exact_build_time_sec,
        prepare_time_sec=exact_insert_time_sec + exact_build_time_sec,
        search_avg_latency_ms=mean(exact_latencies_ms),
        search_p50_latency_ms=percentile(exact_latencies_ms, 50),
        search_p95_latency_ms=percentile(exact_latencies_ms, 95),
        search_p99_latency_ms=percentile(exact_latencies_ms, 99),
        recall_avg=1.0,
        recall_p50=1.0,
        recall_p95=1.0,
    )
    return exact_benchmark_row, exact_result_ids_by_query


def run_ivf_benchmark(
        records: list[VectorRecord],
        query_vectors: list[np.ndarray],
        exact_result_ids_by_query: list[list[str]],
        buffer_size: int,
        top_k: int,
        nlist: int,
        nprobe: int) -> BenchmarkRow:
    print(f"\nBenchmarking IVF nlist={nlist}, nprobe={nprobe}")
    store = IVFVectorStore(
        nlist=nlist,
        nprobe=nprobe,
        buffer_size=buffer_size,
    )
    insert_time_sec = insert_records(store, records)
    build_time_sec = maybe_build_store(store)
    for query_vector in query_vectors[: min(5, len(query_vectors))]:
        store.search(query_vector=query_vector, top_k=top_k)
    latencies_ms, recalls, diag = evaluate_store(
        store=store,
        query_vectors=query_vectors,
        exact_result_ids_by_query=exact_result_ids_by_query,
        top_k=top_k,
    )
    benchmark_row = BenchmarkRow(
        store="ivf",
        store_parameters=params_to_json(
            {
                "nlist": nlist,
                "nprobe": nprobe,
            }
        ),
        records=len(records),
        queries=len(query_vectors),
        insert_time_sec=insert_time_sec,
        build_time_sec=build_time_sec,
        prepare_time_sec=insert_time_sec + build_time_sec,
        search_avg_latency_ms=mean(latencies_ms),
        search_p50_latency_ms=percentile(latencies_ms, 50),
        search_p95_latency_ms=percentile(latencies_ms, 95),
        search_p99_latency_ms=percentile(latencies_ms, 99),
        recall_avg=mean(recalls),
        recall_p50=percentile(recalls, 50),
        recall_p95=percentile(recalls, 95),
        diag_avg_distance_computations=diag.distance_computations,
        diag_avg_visited_nodes=diag.visited_nodes,
        diag_avg_graph_hops=diag.graph_hops,
        diag_avg_layers_traversed=diag.layers_traversed,
        diag_avg_clusters_scanned=diag.clusters_scanned,
        diag_avg_vectors_scanned=diag.vectors_scanned,
    )
    return benchmark_row




def run_flat_nsw_benchmark(
        records: list[VectorRecord],
        query_vectors: list[np.ndarray],
        exact_result_ids_by_query: list[list[str]],
        top_k: int,
        m:  int,
        ef_search: int) -> BenchmarkRow:
    print(f"\nBenchmarking Flat NSW m={m}, ef_search={ef_search}")
    store = FlatNSWVectorStore(
        m=m,
        ef_search=ef_search,
    )
    insert_time_sec = insert_records(store, records)
    build_time_sec = 0.0
    for query_vector in query_vectors[: min(5, len(query_vectors))]:
        store.search(query_vector=query_vector, top_k=top_k)
    latencies_ms, recalls, diag = evaluate_store(
        store=store,
        query_vectors=query_vectors,
        exact_result_ids_by_query=exact_result_ids_by_query,
        top_k=top_k,
    )
    flat_nsw_benchmark_row = BenchmarkRow(
        store="flat_nsw",
        store_parameters=params_to_json(
            {
                "m": m,
                "ef_search": ef_search,
            }
        ),
        records=len(records),
        queries=len(query_vectors),
        insert_time_sec=insert_time_sec,
        build_time_sec=build_time_sec,
        prepare_time_sec=insert_time_sec + build_time_sec,
        search_avg_latency_ms=mean(latencies_ms),
        search_p50_latency_ms=percentile(latencies_ms, 50),
        search_p95_latency_ms=percentile(latencies_ms, 95),
        search_p99_latency_ms=percentile(latencies_ms, 99),
        recall_avg=mean(recalls),
        recall_p50=percentile(recalls, 50),
        recall_p95=percentile(recalls, 95),
        diag_avg_distance_computations=diag.distance_computations,
        diag_avg_visited_nodes=diag.visited_nodes,
        diag_avg_graph_hops=diag.graph_hops,
        diag_avg_layers_traversed=diag.layers_traversed,
        diag_avg_clusters_scanned=diag.clusters_scanned,
        diag_avg_vectors_scanned=diag.vectors_scanned,
    )
    return flat_nsw_benchmark_row


def run_hnsw_benchmark(
        records: list[VectorRecord],
        query_vectors: list[np.ndarray],
        exact_result_ids_by_query: list[list[str]],
        top_k: int,
        m: int,
        ef_construction: int,
        ef_search: int,
        level_multiplier: float) -> BenchmarkRow:

    print(
        f"\nBenchmarking HNSW "
        f"m={m}, ef_construction={ef_construction}, "
        f"ef_search={ef_search}, level_multiplier={level_multiplier}"
    )

    store = HNSWVectorStore(
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
        level_multiplier=level_multiplier,
    )

    insert_time_sec = insert_records(store, records)
    build_time_sec = maybe_build_store(store)

    for query_vector in query_vectors[: min(5, len(query_vectors))]:
        store.search(query_vector=query_vector, top_k=top_k)

    latencies_ms, recalls, diag = evaluate_store(
        store=store,
        query_vectors=query_vectors,
        exact_result_ids_by_query=exact_result_ids_by_query,
        top_k=top_k,
    )

    return BenchmarkRow(
        store="hnsw",
        store_parameters=params_to_json(
            {
                "m": m,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
                "level_multiplier": level_multiplier,
            }
        ),
        records=len(records),
        queries=len(query_vectors),
        insert_time_sec=insert_time_sec,
        build_time_sec=build_time_sec,
        prepare_time_sec=insert_time_sec + build_time_sec,
        search_avg_latency_ms=mean(latencies_ms),
        search_p50_latency_ms=percentile(latencies_ms, 50),
        search_p95_latency_ms=percentile(latencies_ms, 95),
        search_p99_latency_ms=percentile(latencies_ms, 99),
        recall_avg=mean(recalls),
        recall_p50=percentile(recalls, 50),
        recall_p95=percentile(recalls, 95),
        diag_avg_distance_computations=diag.distance_computations,
        diag_avg_visited_nodes=diag.visited_nodes,
        diag_avg_graph_hops=diag.graph_hops,
        diag_avg_layers_traversed=diag.layers_traversed,
        diag_avg_clusters_scanned=diag.clusters_scanned,
        diag_avg_vectors_scanned=diag.vectors_scanned,
    )

if __name__ == "__main__":
    main()