import argparse
import csv
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
from tqdm import tqdm

from benchmarks.benchmark_helpers import (
    create_embedding_model,
    embed_texts,
    insert_records,
    maybe_build_store,
    percentile,
)
from vectordb.models import VectorRecord
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore
from vectordb.stores.flat_nsw_inmem import FlatNSWVectorStore
from vectordb.stores.hnsw_inmem import HNSWVectorStore
from vectordb.stores.ivf_inmem import IVFVectorStore


@dataclass(frozen=True)
class QueryEvaluation:
    query_id: str
    precision: float
    recall: float
    hit_rate: float
    mrr: float
    ndcg: float
    latency_ms: float
    retrieved_ids: list[str]
    relevant_ids: list[str]


@dataclass(frozen=True)
class RetrievalQualityBenchmarkRow:
    store: str
    records: int
    queries: int
    top_k: int
    insert_time_sec: float
    build_time_sec: float
    prepare_time_sec: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


def load_corpus(dataset_dir: Path) -> dict[str, str]:
    corpus = {}

    with open(dataset_dir / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["id"]] = row["text"]

    return corpus


def load_queries(dataset_dir: Path) -> dict[str, str]:
    queries = {}

    with open(dataset_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[row["id"]] = row["text"]

    return queries


def load_qrels(dataset_dir: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}

    with open(dataset_dir / "qrels.tsv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            query_id = row["query_id"]
            doc_id = row["doc_id"]
            relevance = int(row["relevance"])

            qrels.setdefault(query_id, {})[doc_id] = relevance

    return qrels


def load_dataset(dataset_dir: Path) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, int]],
]:
    return (
        load_corpus(dataset_dir),
        load_queries(dataset_dir),
        load_qrels(dataset_dir),
    )


def build_records(
    corpus_ids: list[str],
    corpus: dict[str, str],
    corpus_vectors: list[np.ndarray]) -> list[VectorRecord]:

    records: list[VectorRecord] = []

    for i, (doc_id, vector) in enumerate(zip(corpus_ids, corpus_vectors)):
        records.append(
            VectorRecord(
                id=doc_id,
                vector=vector,
                text=corpus[doc_id],
                metadata={
                    "doc_id": doc_id,
                    "corpus_index": i,
                },
            )
        )

    return records


ALL_STORES = ["buffered", "ivf", "flat_nsw", "hnsw"]


def create_store(store_name: str, args):
    if store_name == "buffered":
        return BufferedMatrixInMemVectorStore(buffer_size=args.buffer_size)

    if store_name == "ivf":
        return IVFVectorStore(
            nlist=args.ivf_nlist,
            nprobe=args.ivf_nprobe,
            buffer_size=args.buffer_size,
        )

    if store_name == "flat_nsw":
        return FlatNSWVectorStore(
            m=args.nsw_m,
            ef_search=args.nsw_ef_search,
        )

    if store_name == "hnsw":
        return HNSWVectorStore(
            m=args.hnsw_m,
            ef_construction=args.hnsw_ef_construction,
            ef_search=args.hnsw_ef_search,
            level_multiplier=args.hnsw_level_multiplier,
        )

    raise ValueError(f"Unknown store: {store_name}")


def relevant_doc_ids(qrels_for_query: dict[str, int]) -> set[str]:
    return {doc_id for doc_id, relevance in qrels_for_query.items() if relevance > 0}


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0

    retrieved_relevant_ids = set(retrieved_ids[:k]) & relevant_ids
    return len(retrieved_relevant_ids) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0

    retrieved_relevant_ids = set(retrieved_ids[:k]) & relevant_ids
    return len(retrieved_relevant_ids) / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    retrieved_relevant_ids = set(retrieved_ids[:k]) & relevant_ids
    return 1.0 if retrieved_relevant_ids else 0.0


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids: list[str], qrels_for_query: dict[str, int], k: int) -> float:
    dcg = 0.0

    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        relevance = qrels_for_query.get(doc_id, 0)
        dcg += (2**relevance - 1) / np.log2(rank + 1)

    ideal_relevances = sorted(qrels_for_query.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, start=1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(
        store,
        query_id: str,
        query_vector: np.ndarray,
        qrels_for_query: dict[str, int],
        top_k: int,
) -> QueryEvaluation:
    start = time.perf_counter()
    results = store.search(query_vector=query_vector, top_k=top_k)
    end = time.perf_counter()

    retrieved_ids = [result.record.id for result in results]
    relevant_ids = relevant_doc_ids(qrels_for_query)

    return QueryEvaluation(
        query_id=query_id,
        precision=precision_at_k(retrieved_ids, relevant_ids, top_k),
        recall=recall_at_k(retrieved_ids, relevant_ids, top_k),
        hit_rate=hit_rate_at_k(retrieved_ids, relevant_ids, top_k),
        mrr=mrr_at_k(retrieved_ids, relevant_ids, top_k),
        ndcg=ndcg_at_k(retrieved_ids, qrels_for_query, top_k),
        latency_ms=(end - start) * 1000,
        retrieved_ids=retrieved_ids,
        relevant_ids=sorted(relevant_ids),
    )


def aggregate_results(
        store_name: str,
        records: list[VectorRecord],
        query_evaluations: list[QueryEvaluation],
        top_k: int,
        insert_time_sec: float,
        build_time_sec: float,
) -> RetrievalQualityBenchmarkRow:
    latencies = [result.latency_ms for result in query_evaluations]

    return RetrievalQualityBenchmarkRow(
        store=store_name,
        records=len(records),
        queries=len(query_evaluations),
        top_k=top_k,
        insert_time_sec=insert_time_sec,
        build_time_sec=build_time_sec,
        prepare_time_sec=insert_time_sec + build_time_sec,
        avg_latency_ms=mean(latencies),
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        p99_latency_ms=percentile(latencies, 99),
        precision_at_k=mean(result.precision for result in query_evaluations),
        recall_at_k=mean(result.recall for result in query_evaluations),
        hit_rate_at_k=mean(result.hit_rate for result in query_evaluations),
        mrr_at_k=mean(result.mrr for result in query_evaluations),
        ndcg_at_k=mean(result.ndcg for result in query_evaluations),
    )


def print_summary(row: RetrievalQualityBenchmarkRow):
    print("\nRetrieval quality benchmark")
    print("---------------------------")
    print(f"store          : {row.store}")
    print(f"records        : {row.records}")
    print(f"queries        : {row.queries}")
    print(f"top_k          : {row.top_k}")

    print("\nPreparation")
    print("-----------")
    print(f"insert time    : {row.insert_time_sec:.4f} sec")
    print(f"build time     : {row.build_time_sec:.4f} sec")
    print(f"prepare time   : {row.prepare_time_sec:.4f} sec")

    print("\nQuality")
    print("-------")
    print(f"Precision@{row.top_k:<3}: {row.precision_at_k:.4f}")
    print(f"Recall@{row.top_k:<6}: {row.recall_at_k:.4f}")
    print(f"HitRate@{row.top_k:<5}: {row.hit_rate_at_k:.4f}")
    print(f"MRR@{row.top_k:<9}: {row.mrr_at_k:.4f}")
    print(f"nDCG@{row.top_k:<8}: {row.ndcg_at_k:.4f}")

    print("\nLatency")
    print("-------")
    print(f"avg latency    : {row.avg_latency_ms:.4f} ms")
    print(f"p50 latency    : {row.p50_latency_ms:.4f} ms")
    print(f"p95 latency    : {row.p95_latency_ms:.4f} ms")
    print(f"p99 latency    : {row.p99_latency_ms:.4f} ms")


def write_summary_csv(
        output_csv: str,
        rows: list[RetrievalQualityBenchmarkRow],
):
    output_path = Path(output_csv)
    row_dicts = [dataclasses.asdict(row) for row in rows]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(row_dicts)

    print(f"\nWrote benchmark summary to: {output_path}")


def write_per_query_csv(
        output_csv: str,
        query_evaluations: list[QueryEvaluation],
):
    output_path = Path(output_csv)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(query_evaluations[0].__dict__.keys()) if query_evaluations else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for evaluation in query_evaluations:
            row = {**evaluation.__dict__, "retrieved_ids": json.dumps(evaluation.retrieved_ids),
                   "relevant_ids": json.dumps(evaluation.relevant_ids)}
            writer.writerow(row)

    print(f"\nWrote per-query benchmark results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval quality using labeled query-document relevance judgments"
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--store",
        choices=["buffered", "ivf", "flat_nsw", "hnsw", "all"],
        default="all",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--buffer-size", type=int, default=1024)

    parser.add_argument(
        "--embedding-model",
        choices=["fake", "sentence-transformer"],
        default="fake",
    )

    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--per-query-output-csv", default=None)

    parser.add_argument("--ivf-nlist", type=int, default=10)
    parser.add_argument("--ivf-nprobe", type=int, default=3)

    parser.add_argument("--nsw-m", type=int, default=8)
    parser.add_argument("--nsw-ef-search", type=int, default=32)

    parser.add_argument("--hnsw-m", type=int, default=8)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    parser.add_argument("--hnsw-ef-search", type=int, default=32)
    parser.add_argument("--hnsw-level-multiplier", type=float, default=1.0)

    args = parser.parse_args()

    stores_to_run = ALL_STORES if args.store == "all" else [args.store]
    multi_store = len(stores_to_run) > 1

    if multi_store and args.per_query_output_csv:
        print("Warning: --per-query-output-csv is ignored when --store is 'all'")

    dataset_dir = Path(args.dataset)

    corpus, queries, qrels = load_dataset(dataset_dir)

    print("\nLoading retrieval quality dataset")
    print("---------------------------------")
    print(f"dataset path  : {dataset_dir}")
    print(f"corpus docs   : {len(corpus)}")
    print(f"queries       : {len(queries)}")
    print(f"qrels queries : {len(qrels)}")

    missing_qrels = sorted(set(queries.keys()) - set(qrels.keys()))
    if missing_qrels:
        raise ValueError(
            f"Missing qrels for {len(missing_qrels)} queries: {missing_qrels[:10]}"
        )

    embedding_model = create_embedding_model(args.embedding_model)

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]

    print("\nEmbedding corpus")
    print("----------------")
    corpus_vectors = embed_texts(
        embedding_model=embedding_model,
        texts=corpus_texts,
        desc="Embedding corpus",
    )

    records = build_records(
        corpus_ids=corpus_ids,
        corpus=corpus,
        corpus_vectors=corpus_vectors,
    )

    query_ids = list(queries.keys())
    query_texts = [queries[query_id] for query_id in query_ids]

    print("\nEmbedding queries")
    print("-----------------")
    query_vectors = embed_texts(
        embedding_model=embedding_model,
        texts=query_texts,
        desc="Embedding queries",
    )

    summary_rows: list[RetrievalQualityBenchmarkRow] = []
    last_query_evaluations: list[QueryEvaluation] = []

    for store_name in stores_to_run:
        print(f"\nPreparing store: {store_name}")
        print("---------------")
        store = create_store(store_name, args)
        insert_time_sec = insert_records(store, records)
        build_time_sec = maybe_build_store(store)

        print(f"\nEvaluating queries: {store_name}")
        print("------------------")

        query_evaluations: list[QueryEvaluation] = []

        for query_id, query_vector in tqdm(
                list(zip(query_ids, query_vectors)),
                desc="Retrieval quality",
        ):
            query_evaluations.append(
                evaluate_query(
                    store=store,
                    query_id=query_id,
                    query_vector=query_vector,
                    qrels_for_query=qrels[query_id],
                    top_k=args.top_k,
                )
            )

        row = aggregate_results(
            store_name=store_name,
            records=records,
            query_evaluations=query_evaluations,
            top_k=args.top_k,
            insert_time_sec=insert_time_sec,
            build_time_sec=build_time_sec,
        )

        summary_rows.append(row)
        print_summary(row)

        if not multi_store:
            last_query_evaluations = query_evaluations

    if args.output_csv:
        write_summary_csv(args.output_csv, summary_rows)

    if not multi_store and args.per_query_output_csv:
        write_per_query_csv(args.per_query_output_csv, last_query_evaluations)


if __name__ == "__main__":
    main()