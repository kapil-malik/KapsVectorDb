import argparse
import csv
import dataclasses
import itertools
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
    embed_texts_with_cache,
    insert_records,
    maybe_build_store,
    parse_int_list,
    percentile,
)
from vectordb.models import VectorRecord
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore
from vectordb.stores.flat_nsw_inmem import FlatNSWVectorStore
from vectordb.stores.flat_nsw_inmem_v2 import FlatNSWV2VectorStore
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
    store_parameters: str
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
            doc_id = row["id"]
            if doc_id in corpus:
                raise ValueError(f"Duplicate doc_id in corpus.jsonl: {doc_id!r}")
            corpus[doc_id] = row["text"]

    return corpus


def load_queries(dataset_dir: Path) -> dict[str, str]:
    queries = {}

    with open(dataset_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            query_id = row["id"]
            if query_id in queries:
                raise ValueError(f"Duplicate query_id in queries.jsonl: {query_id!r}")
            queries[query_id] = row["text"]

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


def validate_dataset(
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> None:
    unknown_qrel_queries = sorted(set(qrels.keys()) - set(queries.keys()))
    if unknown_qrel_queries:
        raise ValueError(
            f"qrels reference {len(unknown_qrel_queries)} query_id(s) not in queries.jsonl: "
            f"{unknown_qrel_queries[:10]}"
        )

    missing_qrels = sorted(set(queries.keys()) - set(qrels.keys()))
    if missing_qrels:
        raise ValueError(
            f"Missing qrels for {len(missing_qrels)} queries: {missing_qrels[:10]}"
        )

    all_qrel_doc_ids = {doc_id for qrel in qrels.values() for doc_id in qrel}
    unknown_doc_ids = sorted(all_qrel_doc_ids - set(corpus.keys()))
    if unknown_doc_ids:
        raise ValueError(
            f"qrels reference {len(unknown_doc_ids)} doc_id(s) not in corpus.jsonl: "
            f"{unknown_doc_ids[:10]}"
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


ALL_STORES = ["buffered", "ivf", "flat_nsw_v2", "hnsw"]


def param_grid(store_name: str, args) -> list[dict]:
    if store_name == "buffered":
        return [{"buffer_size": args.buffer_size}]

    if store_name == "ivf":
        return [
            {"buffer_size": args.buffer_size, "nlist": nlist, "nprobe": nprobe}
            for nlist, nprobe in itertools.product(args.ivf_nlists, args.ivf_nprobes)
            if nprobe <= nlist
        ]

    if store_name == "flat_nsw":
        return [
            {"ef_search": ef_search, "m": m}
            for m, ef_search in itertools.product(args.nsw_ms, args.nsw_ef_searches)
        ]

    if store_name == "flat_nsw_v2":
        return [
            {"m": m, "ef_construction": ef_construction, "ef_search": ef_search}
            for m, ef_construction, ef_search in itertools.product(
                args.nsw_ms, args.nsw_ef_constructions, args.nsw_ef_searches
            )
        ]

    if store_name == "hnsw":
        return [
            {"ef_construction": ef_construction, "ef_search": ef_search,
             "level_multiplier": args.hnsw_level_multiplier, "m": m}
            for m, ef_construction, ef_search in itertools.product(
                args.hnsw_ms, args.hnsw_ef_constructions, args.hnsw_ef_searches
            )
        ]

    raise ValueError(f"Unknown store: {store_name}")


def create_store(store_name: str, params: dict):
    if store_name == "buffered":
        return BufferedMatrixInMemVectorStore(buffer_size=params["buffer_size"])

    if store_name == "ivf":
        return IVFVectorStore(
            nlist=params["nlist"],
            nprobe=params["nprobe"],
            buffer_size=params["buffer_size"],
        )

    if store_name == "flat_nsw":
        return FlatNSWVectorStore(
            m=params["m"],
            ef_search=params["ef_search"],
        )

    if store_name == "flat_nsw_v2":
        return FlatNSWV2VectorStore(
            m=params["m"],
            ef_construction=params["ef_construction"],
            ef_search=params["ef_search"],
        )

    if store_name == "hnsw":
        return HNSWVectorStore(
            m=params["m"],
            ef_construction=params["ef_construction"],
            ef_search=params["ef_search"],
            level_multiplier=params["level_multiplier"],
        )

    raise ValueError(f"Unknown store: {store_name}")


def _build_key(store_name: str, params: dict) -> str:
    """Cache key covering only construction-time parameters.

    flat_nsw / flat_nsw_v2 / hnsw: ef_search is search-time only, so exclude it.
    The same built store can be reused across ef_search sweep values by patching
    the attribute directly — no rebuild needed.
    """
    if store_name in ("flat_nsw", "flat_nsw_v2", "hnsw"):
        build_p = {k: v for k, v in params.items() if k != "ef_search"}
    else:
        build_p = params
    return store_name + "::" + json.dumps(build_p, sort_keys=True)


def _apply_search_params(store, store_name: str, params: dict) -> None:
    """Patch search-time parameters onto an already-built store (no rebuild)."""
    if store_name in ("flat_nsw", "flat_nsw_v2", "hnsw"):
        store.ef_search = params["ef_search"]


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
        store_parameters: str,
        records: list[VectorRecord],
        query_evaluations: list[QueryEvaluation],
        top_k: int,
        insert_time_sec: float,
        build_time_sec: float,
) -> RetrievalQualityBenchmarkRow:
    latencies = [result.latency_ms for result in query_evaluations]

    return RetrievalQualityBenchmarkRow(
        store=store_name,
        store_parameters=store_parameters,
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
    print(f"parameters     : {row.store_parameters}")
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
        choices=["buffered", "ivf", "flat_nsw", "flat_nsw_v2", "hnsw", "all"],
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
    parser.add_argument("--embedding-cache-dir", default=None,
                        help="Directory for caching corpus and query embeddings as .npy files")
    parser.add_argument("--max-eval-queries", type=int, default=None,
                        help="Evaluate only the first N queries (applied after dataset validation)")

    parser.add_argument("--ivf-nlist", default="10",
                        help="IVF nlist value(s), comma-separated for sweep (e.g. '10,50,100')")
    parser.add_argument("--ivf-nprobe", default="3",
                        help="IVF nprobe value(s), comma-separated for sweep (e.g. '1,3,5')")

    parser.add_argument("--nsw-m", default="8",
                        help="FlatNSW M value(s), comma-separated for sweep (e.g. '8,16')")
    parser.add_argument("--nsw-ef-search", default="32",
                        help="FlatNSW ef_search value(s), comma-separated for sweep")
    parser.add_argument("--nsw-ef-construction", default="64",
                        help="FlatNSW v2 ef_construction value(s), comma-separated for sweep (e.g. '32,64,128')")

    parser.add_argument("--hnsw-m", default="8",
                        help="HNSW M value(s), comma-separated for sweep (e.g. '8,16')")
    parser.add_argument("--hnsw-ef-construction", default="64",
                        help="HNSW ef_construction value(s), comma-separated for sweep")
    parser.add_argument("--hnsw-ef-search", default="32",
                        help="HNSW ef_search value(s), comma-separated for sweep")
    parser.add_argument("--hnsw-level-multiplier", type=float, default=1.0)

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError(f"--top-k must be > 0, got {args.top_k}")

    args.ivf_nlists = parse_int_list(args.ivf_nlist)
    args.ivf_nprobes = parse_int_list(args.ivf_nprobe)
    args.nsw_ms = parse_int_list(args.nsw_m)
    args.nsw_ef_searches = parse_int_list(args.nsw_ef_search)
    args.nsw_ef_constructions = parse_int_list(args.nsw_ef_construction)
    args.hnsw_ms = parse_int_list(args.hnsw_m)
    args.hnsw_ef_constructions = parse_int_list(args.hnsw_ef_construction)
    args.hnsw_ef_searches = parse_int_list(args.hnsw_ef_search)

    stores_to_run = ALL_STORES if args.store == "all" else [args.store]
    total_configs = sum(len(param_grid(s, args)) for s in stores_to_run)
    is_single_config = total_configs == 1

    if not is_single_config and args.per_query_output_csv:
        print("Warning: --per-query-output-csv is ignored when running multiple configurations "
              "(use a single store with single-value parameters to enable it)")

    cache_dir = Path(args.embedding_cache_dir) if args.embedding_cache_dir else None
    dataset_dir = Path(args.dataset)

    corpus, queries, qrels = load_dataset(dataset_dir)

    print("\nLoading retrieval quality dataset")
    print("---------------------------------")
    print(f"dataset path  : {dataset_dir}")
    print(f"corpus docs   : {len(corpus)}")
    print(f"queries       : {len(queries)}")
    print(f"qrels queries : {len(qrels)}")

    validate_dataset(corpus, queries, qrels)

    query_ids = list(queries.keys())
    if args.max_eval_queries is not None and args.max_eval_queries < len(query_ids):
        query_ids = query_ids[:args.max_eval_queries]
        print(f"eval queries  : {len(query_ids)} (capped by --max-eval-queries)")

    embedding_model = create_embedding_model(args.embedding_model)

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]

    print("\nEmbedding corpus")
    print("----------------")
    corpus_vectors = embed_texts_with_cache(
        embedding_model=embedding_model,
        model_type=args.embedding_model,
        ids=corpus_ids,
        texts=corpus_texts,
        desc="Embedding corpus",
        cache_dir=cache_dir,
        cache_prefix="corpus",
    )

    records = build_records(
        corpus_ids=corpus_ids,
        corpus=corpus,
        corpus_vectors=corpus_vectors,
    )

    query_texts = [queries[qid] for qid in query_ids]

    print("\nEmbedding queries")
    print("-----------------")
    query_vectors = embed_texts_with_cache(
        embedding_model=embedding_model,
        model_type=args.embedding_model,
        ids=query_ids,
        texts=query_texts,
        desc="Embedding queries",
        cache_dir=cache_dir,
        cache_prefix="query",
    )

    summary_rows: list[RetrievalQualityBenchmarkRow] = []
    last_query_evaluations: list[QueryEvaluation] = []
    built_store_cache: dict[str, tuple] = {}

    for store_name in stores_to_run:
        for params in param_grid(store_name, args):
            cache_key = _build_key(store_name, params)

            if cache_key not in built_store_cache:
                print(f"\nPreparing store: {store_name}")
                print("---------------")
                store = create_store(store_name, params)
                insert_time_sec = insert_records(store, records)
                build_time_sec = maybe_build_store(store)
                built_store_cache[cache_key] = (store, insert_time_sec, build_time_sec)
            else:
                store, insert_time_sec, build_time_sec = built_store_cache[cache_key]
                _apply_search_params(store, store_name, params)
                print(f"\nReusing store: {store_name} (ef_search → {params['ef_search']})")
                print("-------------")

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
                store_parameters=json.dumps(params, sort_keys=True),
                records=records,
                query_evaluations=query_evaluations,
                top_k=args.top_k,
                insert_time_sec=insert_time_sec,
                build_time_sec=build_time_sec,
            )

            summary_rows.append(row)
            print_summary(row)

            if is_single_config:
                last_query_evaluations = query_evaluations

    if args.output_csv:
        write_summary_csv(args.output_csv, summary_rows)

    if is_single_config and args.per_query_output_csv:
        write_per_query_csv(args.per_query_output_csv, last_query_evaluations)


if __name__ == "__main__":
    main()