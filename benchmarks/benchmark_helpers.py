import json
import time
from pathlib import Path
from statistics import mean

import numpy as np
from tqdm import tqdm

from vectordb.embeddings.fake import FakeHashEmbeddingModel
from vectordb.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from vectordb.models import VectorRecord


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def embed_texts_with_cache(
    embedding_model,
    model_type: str,
    ids: list[str],
    texts: list[str],
    desc: str,
    cache_dir: Path | None,
    cache_prefix: str,
) -> list[np.ndarray]:
    if cache_dir is None:
        return embed_texts(embedding_model, texts, desc)

    npy_path = cache_dir / f"{cache_prefix}_embeddings.npy"
    meta_path = cache_dir / f"{cache_prefix}_embeddings.meta.json"

    if npy_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            invalid_reason = None
            if meta.get("model_type") != model_type:
                invalid_reason = f"model changed ({meta.get('model_type')!r} → {model_type!r})"
            elif meta.get("ids") != ids:
                invalid_reason = "IDs changed or reordered"
            else:
                matrix = np.load(npy_path)
                if matrix.shape[0] != len(texts):
                    invalid_reason = f"vector count mismatch ({matrix.shape[0]} vs {len(texts)})"
                else:
                    print(f"cache hit   : {npy_path}")
                    return [matrix[i] for i in range(len(matrix))]
            print(f"cache stale : {invalid_reason}, recomputing ...")
        except Exception as e:
            print(f"cache error : {e}, recomputing ...")
    else:
        print(f"cache miss  : no cache found, will save to {npy_path}")

    vectors = embed_texts(embedding_model, texts, desc)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, np.array(vectors))
    meta = {"model_type": model_type, "num_texts": len(texts), "ids": ids}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"cache saved : {npy_path}")
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
