import argparse
import time
from statistics import mean

from tqdm import tqdm

from vectordb.embeddings.fake import FakeHashEmbeddingModel
from vectordb.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from vectordb.models import VectorRecord
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore
from vectordb.stores.ivf_inmem import IVFVectorStore
from pathlib import Path

def load_lines(file_path: str) -> list[str]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if stripped:
                lines.append(stripped)

    if not lines:
        raise ValueError(f"No non-empty lines found in {file_path}")

    return lines

def percentile(values: list[float], p: float) -> float:
    sorted_values = sorted(values)
    index = int((p / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


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

def create_embedding_model(model_type: str):
    if model_type == "fake":
        return FakeHashEmbeddingModel(dimension=384)

    if model_type == "sentence-transformer":
        return SentenceTransformerEmbeddingModel()

    raise ValueError(f"Unknown model type: {model_type}")


def embed_texts(embedding_model, texts: list[str], desc: str):
    vectors = []

    for text in tqdm(texts, desc=desc):
        vectors.append(embedding_model.embed(text))

    return vectors


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


def insert_records(store, records: list[VectorRecord]) -> None:
    for record in tqdm(records, desc=f"Inserting into {store.__class__.__name__}"):
        store.insert(record)


def search_with_latency(store, query_vector, top_k: int):
    start = time.perf_counter()
    results = store.search(query_vector=query_vector, top_k=top_k)
    end = time.perf_counter()

    return results, (end - start) * 1000


def recall_at_k(exact_ids: list[str], candidate_ids: list[str], k: int) -> float:
    exact_top_k = set(exact_ids[:k])
    candidate_top_k = set(candidate_ids[:k])

    if not exact_top_k:
        return 0.0

    return len(exact_top_k.intersection(candidate_top_k)) / k


def print_latency_stats(title: str, latencies_ms: list[float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"avg latency   : {mean(latencies_ms):.4f} ms")
    print(f"p50 latency   : {percentile(latencies_ms, 50):.4f} ms")
    print(f"p95 latency   : {percentile(latencies_ms, 95):.4f} ms")
    print(f"p99 latency   : {percentile(latencies_ms, 99):.4f} ms")


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

    insert_records(exact_store, records)
    insert_records(ivf_store, records)

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