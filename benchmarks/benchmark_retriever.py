import argparse
import time
from statistics import mean

from tqdm import tqdm

from vectordb.embeddings.fake import FakeHashEmbeddingModel
from vectordb.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from vectordb.retrieval.semantic_text_retriever import SemanticTextRetriever
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore


def percentile(values: list[float], p: float) -> float:
    sorted_values = sorted(values)
    index = int((p / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def generate_synthetic_chunks(num_chunks: int) -> list[str]:
    topics = [
        "running and cardiovascular fitness",
        "vector databases and embeddings",
        "distributed systems and Kafka",
        "strength training and recovery",
        "retrieval augmented generation",
    ]

    chunks = []

    for i in range(num_chunks):
        topic = topics[i % len(topics)]
        chunks.append(
            f"This is synthetic chunk {i}. It discusses {topic}. "
            f"The purpose is to benchmark semantic text retrieval."
        )

    return chunks


def create_embedding_model(model_type: str):
    if model_type == "fake":
        return FakeHashEmbeddingModel(dimension=384)

    if model_type == "sentence-transformer":
        return SentenceTransformerEmbeddingModel()

    raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embedding-model",
        choices=["fake", "sentence-transformer"],
        default="fake",
    )

    args = parser.parse_args()

    embedding_model = create_embedding_model(args.embedding_model)
    vector_store = BufferedMatrixInMemVectorStore(buffer_size=1024)

    retriever = SemanticTextRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
    )

    chunks = generate_synthetic_chunks(args.chunks)

    start = time.perf_counter()
    retriever.add_chunks(chunks)
    end = time.perf_counter()

    ingest_time_sec = end - start

    print("\nRetriever ingestion benchmark")
    print("-----------------------------")
    print(f"chunks          : {args.chunks}")
    print(f"embedding model : {args.embedding_model}")
    print(f"total time      : {ingest_time_sec:.4f} sec")
    print(f"chunks/sec      : {args.chunks / ingest_time_sec:.2f}")

    query_latencies_ms = []

    queries = [
        "How do vector databases use embeddings?",
        "What improves running stamina?",
        "How is Kafka used in distributed systems?",
        "What is retrieval augmented generation?",
        "How does strength training help?",
    ]

    for i in tqdm(range(args.queries), desc="Retriever search"):
        query = queries[i % len(queries)]

        start = time.perf_counter()
        retriever.search(query, top_k=args.top_k)
        end = time.perf_counter()

        query_latencies_ms.append((end - start) * 1000)

    print("\nRetriever search benchmark")
    print("--------------------------")
    print(f"queries      : {args.queries}")
    print(f"top_k        : {args.top_k}")
    print(f"avg latency  : {mean(query_latencies_ms):.4f} ms")
    print(f"p50 latency  : {percentile(query_latencies_ms, 50):.4f} ms")
    print(f"p95 latency  : {percentile(query_latencies_ms, 95):.4f} ms")
    print(f"p99 latency  : {percentile(query_latencies_ms, 99):.4f} ms")


if __name__ == "__main__":
    main()