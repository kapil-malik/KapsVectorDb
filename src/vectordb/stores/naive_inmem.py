import numpy as np

from vectordb.distance import cosine_similarity
from vectordb.models import SearchResult, VectorRecord


class NaiveInMemVectorStore:
    def __init__(self):
        self._records: dict[str, VectorRecord] = {}

    def insert(self, record: VectorRecord) -> None:
        if record.id in self._records:
            raise ValueError(f"Record with id '{record.id}' already exists")

        self._records[record.id] = record

    def get(self, record_id: str) -> VectorRecord | None:
        return self._records.get(record_id)

    def delete(self, record_id: str) -> bool:
        if record_id not in self._records:
            return False

        del self._records[record_id]
        return True

    def count(self) -> int:
        return len(self._records)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        results: list[SearchResult] = []

        for record in self._records.values():
            score = cosine_similarity(query_vector, record.vector)
            results.append(SearchResult(record=record, score=score))

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]