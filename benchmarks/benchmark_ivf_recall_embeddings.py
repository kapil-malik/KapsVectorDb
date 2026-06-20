import argparse
import time
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from benchmarks.benchmark_helpers import (
    create_embedding_model,
    embed_texts,
    insert_records,
    load_lines,
    percentile,
    print_latency_stats,
    search_with_latency,
)
from vectordb.models import VectorRecord
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore
from vectordb.stores.ivf_inmem import IVFVectorStore


def generate_chunks_from_topics(
        topics: list[str],
        num_chunks: int,
) -> list[str]:
    chunks = []

    for i in range(num_chunks):
        topic = topics[i % len(topics)]

        chunks.append(
            f"This document discusses {topic}. "
            f"It contains information related to {topic}. "
            f"This is synthetic benchmark chunk {i}."
        )

    return chunks


def generate_queries_from_templates(
        queries: list[str],
        num_queries: int,
) -> list[str]:
    return [
        queries[i % len(queries)]
        for i in range(num_queries)
    ]



def create_records(chunks: list[str], vectors) -> list[VectorRecord]:
    records = []

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        topic_id = i % 5

        records.append(
            VectorRecord(
                id=f"chunk-{i}",
                vector=vector,
                text=chunk,
                metadata={
                    "source": "synthetic",
                    "topic_id": topic_id,
                },
            )
        )

    return records




def recall_at_k(exact_ids: list[str], candidate_ids: list[str], k: int) -> float:
    exact_top_k = set(exact_ids[:k])
    candidate_top_k = set(candidate_ids[:k])

    if not exact_top_k:
        return 0.0

    return len(exact_top_k.intersection(candidate_top_k)) / k


def benchmark_ivf_recall_with_embeddings(
        chunks: list[str],
        query_texts: list[str],
        embedding_model,
        top_k: int,
        nlist: int,
        nprobe: int,
) -> None:
    print("\nEmbedding chunks")
    print("----------------")
    start = time.perf_counter()
    chunk_vectors = embed_texts(embedding_model, chunks, "Embedding chunks")
    end = time.perf_counter()
    chunk_embedding_time_sec = end - start

    print(f"chunk embedding time : {chunk_embedding_time_sec:.4f} sec")
    print(f"chunks/sec           : {len(chunks) / chunk_embedding_time_sec:.2f}")

    records = create_records(chunks, chunk_vectors)

    print("\nEmbedding queries")
    print("-----------------")
    start = time.perf_counter()
    query_vectors = embed_texts(embedding_model, query_texts, "Embedding queries")
    end = time.perf_counter()
    query_embedding_time_sec = end - start

    print(f"query embedding time : {query_embedding_time_sec:.4f} sec")
    print(f"queries/sec          : {len(query_texts) / query_embedding_time_sec:.2f}")

    exact_store = BufferedMatrixInMemVectorStore(buffer_size=1024)
    ivf_store = IVFVectorStore(
        nlist=nlist,
        nprobe=nprobe,
        buffer_size=1024,
    )

    insert_records(exact_store, records, desc=f"Inserting into {exact_store.__class__.__name__}")
    insert_records(ivf_store, records, desc=f"Inserting into {ivf_store.__class__.__name__}")

    print("\nBuilding IVF index")
    print("------------------")
    start = time.perf_counter()
    ivf_store.build()
    end = time.perf_counter()
    build_time_sec = end - start
    print(f"build time : {build_time_sec:.4f} sec")

    exact_latencies_ms: list[float] = []
    ivf_latencies_ms: list[float] = []
    recalls: list[float] = []

    # Warmup
    for query_vector in query_vectors[:5]:
        exact_store.search(query_vector=query_vector, top_k=top_k)
        ivf_store.search(query_vector=query_vector, top_k=top_k)

    for query_vector in tqdm(query_vectors, desc="Comparing exact vs IVF"):
        exact_results, exact_latency_ms = search_with_latency(
            exact_store,
            query_vector,
            top_k,
        )

        ivf_results, ivf_latency_ms = search_with_latency(
            ivf_store,
            query_vector,
            top_k,
        )

        exact_ids = [result.record.id for result in exact_results]
        ivf_ids = [result.record.id for result in ivf_results]

        exact_latencies_ms.append(exact_latency_ms)
        ivf_latencies_ms.append(ivf_latency_ms)
        recalls.append(recall_at_k(exact_ids, ivf_ids, top_k))

    print("\nIVF recall benchmark with embeddings")
    print("------------------------------------")
    print(f"chunks        : {len(chunks)}")
    print(f"queries       : {len(query_texts)}")
    print(f"top_k         : {top_k}")
    print(f"nlist         : {nlist}")
    print(f"nprobe        : {nprobe}")

    print_latency_stats("Exact baseline latency", exact_latencies_ms)
    print_latency_stats("IVF latency", ivf_latencies_ms)

    print(f"\nRecall")
    print("------")
    print(f"avg recall@{top_k} : {mean(recalls):.4f}")
    print(f"p50 recall@{top_k} : {percentile(recalls, 50):.4f}")
    print(f"p95 recall@{top_k} : {percentile(recalls, 95):.4f}")

    speedup = mean(exact_latencies_ms) / mean(ivf_latencies_ms)

    print("\nSummary")
    print("-------")
    print(f"build time     : {build_time_sec:.4f} sec")
    print(f"speedup        : {speedup:.2f}x")
    print(f"recall@{top_k}      : {mean(recalls):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark IVF recall using real text embeddings"
    )

    parser.add_argument("--chunks", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--nlist", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=5)
    parser.add_argument(
        "--embedding-model",
        choices=["fake", "sentence-transformer"],
        default="sentence-transformer",
    )
    parser.add_argument(
        "--topics-file",
        type=str,
        default=str((Path(__file__).parent / "data" / "topics.txt")),
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=str((Path(__file__).parent / "data" / "queries.txt")),
    )

    args = parser.parse_args()

    embedding_model = create_embedding_model(args.embedding_model)

    topics = load_lines(args.topics_file)
    query_templates = load_lines(args.queries_file)

    chunks = generate_chunks_from_topics(
        topics=topics,
        num_chunks=args.chunks,
    )

    query_texts = generate_queries_from_templates(
        queries=query_templates,
        num_queries=args.queries,
    )
    print(f"topics file   : {args.topics_file}")
    print(f"queries file  : {args.queries_file}")
    print(f"topic count   : {len(topics)}")

    benchmark_ivf_recall_with_embeddings(
        chunks=chunks,
        query_texts=query_texts,
        embedding_model=embedding_model,
        top_k=args.top_k,
        nlist=args.nlist,
        nprobe=args.nprobe,
    )


if __name__ == "__main__":
    main()