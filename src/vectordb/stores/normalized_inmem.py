from typing import Any

import numpy as np

from vectordb.distance import dot_similarity
from vectordb.filters import metadata_matches
from vectordb.models import SearchResult, VectorRecord


class NormalizedInMemVectorStore:
    def __init__(self):
        self._records: dict[str, VectorRecord] = {}

    def insert(self, record: VectorRecord) -> None:
        if record.id in self._records:
            raise ValueError(f"Record with id '{record.id}' already exists")

        norm = np.linalg.norm(record.vector)
        if norm == 0:
            raise ValueError("Cannot insert zero vector")

        normalized_vector = (record.vector / norm).astype(np.float32)

        normalized_record = VectorRecord(
            id=record.id,
            vector=normalized_vector,
            text=record.text,
            metadata=record.metadata,
        )
        self._records[record.id] = normalized_record

    def get(self, record_id: str) -> VectorRecord | None:
        return self._records.get(record_id)

    def delete(self, record_id: str) -> bool:
        if record_id not in self._records:
            return False

        del self._records[record_id]
        return True

    def count(self) -> int:
        return len(self._records)

    def search(
            self,
            query_vector: np.ndarray,
            top_k: int = 5,
            filters: dict[str, Any] | None = None) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        results: list[SearchResult] = []

        norm = np.linalg.norm(query_vector)
        if norm == 0:
            raise ValueError("Cannot search with zero vector")

        normalized_query = (query_vector / norm).astype(np.float32)

        for record in self._records.values():
            if not metadata_matches(record, filters):
                continue
            score = dot_similarity(normalized_query, record.vector)
            results.append(SearchResult(record=record, score=score))

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]