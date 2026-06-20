import time
from pathlib import Path
from statistics import mean

import numpy as np
from tqdm import tqdm

from vectordb.embeddings.fake import FakeHashEmbeddingModel
from vectordb.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from vectordb.models import VectorRecord


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def create_embedding_model(model_type: str):
    if model_type == "fake":
        return FakeHashEmbeddingModel(dimension=384)
    if model_type == "sentence-transformer":
        return SentenceTransformerEmbeddingModel()
    raise ValueError(f"Unknown model type: {model_type}")


def embed_texts(embedding_model, texts: list[str], desc: str) -> list[np.ndarray]:
    vectors = []
    for text in tqdm(texts, desc=desc):
        vectors.append(embedding_model.embed(text))
    return vectors


def insert_records(store, records: list[VectorRecord], desc: str = "Inserting records") -> float:
    start = time.perf_counter()
    for record in tqdm(records, desc=desc):
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


def generate_random_vector(dim: int) -> np.ndarray:
    return np.random.random(dim).astype(np.float32)


def generate_records(num_records: int, dim: int, desc: str = "Generating records") -> list[VectorRecord]:
    records = []
    for i in tqdm(range(num_records), desc=desc):
        records.append(
            VectorRecord(
                id=f"record-{i}",
                vector=generate_random_vector(dim),
                text=f"Synthetic text for record {i}",
                metadata={"source": "synthetic"},
            )
        )
    return records


def search_with_latency(store, query_vector: np.ndarray, top_k: int):
    start = time.perf_counter()
    results = store.search(query_vector=query_vector, top_k=top_k)
    end = time.perf_counter()
    return results, (end - start) * 1000


def load_lines(file_path: str) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No non-empty lines found in {file_path}")
    return lines


def print_latency_stats(title: str, latencies_ms: list[float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"avg latency   : {mean(latencies_ms):.4f} ms")
    print(f"p50 latency   : {percentile(latencies_ms, 50):.4f} ms")
    print(f"p95 latency   : {percentile(latencies_ms, 95):.4f} ms")
    print(f"p99 latency   : {percentile(latencies_ms, 99):.4f} ms")
