from typing import Any

import heapq

import numpy as np

from vectordb.models import SearchResult, VectorRecord
from vectordb.filters import metadata_matches

class BufferedMatrixInMemVectorStore:
    """
    Matrix-backed in-memory vector store with buffered inserts.

    Design:
    - Main vectors live in a NumPy matrix.
    - New inserts go into an in-memory buffer first.
    - Once buffer reaches threshold, we flush buffer into the matrix in one batch.

    This avoids np.vstack on every insert.
    """

    def __init__(self, buffer_size: int = 1024):
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        self._buffer_size = buffer_size

        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []

        self._buffer_vectors: list[np.ndarray] = []
        self._buffer_ids: list[str] = []

        self._records: dict[str, VectorRecord] = {}
        self._dim: int | None = None

    def insert(self, record: VectorRecord) -> None:
        if record.id in self._records:
            raise ValueError(f"Record with id '{record.id}' already exists")

        if record.vector.ndim != 1:
            raise ValueError("Vector must be one-dimensional")

        if record.vector.size == 0:
            raise ValueError("Vector must not be empty")

        if self._dim is None:
            self._dim = record.vector.size
        elif record.vector.size != self._dim:
            raise ValueError(
                f"Vector dimension mismatch. Expected {self._dim}, got {record.vector.size}"
            )

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
        self._buffer_ids.append(record.id)
        self._buffer_vectors.append(normalized_vector)

        if len(self._buffer_vectors) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer_vectors:
            return

        batch_matrix = np.vstack(self._buffer_vectors).astype(np.float32)

        if self._vectors is None:
            self._vectors = batch_matrix
        else:
            self._vectors = np.vstack([self._vectors, batch_matrix])

        self._ids.extend(self._buffer_ids)

        self._buffer_vectors.clear()
        self._buffer_ids.clear()

    def get(self, record_id: str) -> VectorRecord | None:
        return self._records.get(record_id)

    def delete(self, record_id: str) -> bool:
        """
        Tombstone-style delete.

        We remove the record from _records.
        Matrix/buffer entries are physically cleaned later during compaction.
        """
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

        if self.count() == 0:
            return []

        if query_vector.ndim != 1:
            raise ValueError("Query vector must be one-dimensional")

        if self._dim is not None and query_vector.size != self._dim:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self._dim}, got {query_vector.size}"
            )

        norm = np.linalg.norm(query_vector)
        if norm == 0:
            raise ValueError("Cannot search with zero vector")

        normalized_query = (query_vector / norm).astype(np.float32)

        candidates: list[tuple[float, str]] = []
        candidate_multiplier = 10 if filters else 4

        # Search main matrix.
        if self._vectors is not None and len(self._ids) > 0:
            scores = self._vectors @ normalized_query

            # Get more than top_k to tolerate deleted/tombstoned records.
            candidate_count = min(len(scores), top_k * candidate_multiplier)
            top_indices = np.argpartition(-scores, candidate_count - 1)[:candidate_count]

            for index in top_indices:
                record_id = self._ids[index]
                record = self._records.get(record_id)

                if record is None:
                    continue

                if not metadata_matches(record, filters):
                    continue

                candidates.append((float(scores[index]), record_id))

        # Search pending buffer.
        if self._buffer_vectors:
            buffer_matrix = np.vstack(self._buffer_vectors).astype(np.float32)
            buffer_scores = buffer_matrix @ normalized_query

            candidate_count = min(len(buffer_scores), top_k * candidate_multiplier)
            top_indices = np.argpartition(-buffer_scores, candidate_count - 1)[:candidate_count]

            for index in top_indices:
                record_id = self._buffer_ids[index]
                record = self._records.get(record_id)

                if record is None:
                    continue

                if not metadata_matches(record, filters):
                    continue

                candidates.append((float(buffer_scores[index]), record_id))

        top_candidates = heapq.nlargest(top_k, candidates, key=lambda item: item[0])

        return [
            SearchResult(
                record=self._records[record_id],
                score=score,
            )
            for score, record_id in top_candidates
        ]