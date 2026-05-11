import numpy as np

from vectordb.models import SearchResult, VectorRecord


class MatrixBackedInMemVectorStore:
    """
    In-memory vector store backed by a dense NumPy matrix.

    Important idea:
    - Vectors are stored in one contiguous matrix.
    - Search computes all dot products in one vectorized operation.

    This store normalizes vectors at insert time.
    Therefore cosine similarity becomes dot product.
    """

    def __init__(self):
        self._vectors: np.ndarray | None = None
        self._records: dict[str, VectorRecord] = {}
        self._ids: list[str] = []
        self._id_to_index: dict[str, int] = {}
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

        new_row = normalized_vector.reshape(1, -1)

        if self._vectors is None:
            self._vectors = new_row
        else:
            self._vectors = np.vstack([self._vectors, new_row])

        index = len(self._ids)
        self._ids.append(record.id)
        self._id_to_index[record.id] = index
        self._records[record.id] = normalized_record

    def get(self, record_id: str) -> VectorRecord | None:
        return self._records.get(record_id)

    def delete(self, record_id: str) -> bool:
        """
        Simple delete using tombstone semantics.

        We remove it from records, but do not physically remove the row
        from the matrix because deleting a row from a NumPy matrix is expensive.

        This mirrors a real DB idea:
        - mark deleted now
        - compact later
        """
        if record_id not in self._records:
            return False

        del self._records[record_id]
        return True

    def count(self) -> int:
        return len(self._records)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self._vectors is None or len(self._ids) == 0:
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

        scores = self._vectors @ normalized_query

        # If records were deleted, some matrix rows may no longer be active.
        # For now, filter those out after scoring.
        active_results: list[SearchResult] = []

        # np.argsort returns indices in ascending order.
        # We use negative scores to get descending order.
        sorted_indices = np.argsort(-scores)

        for index in sorted_indices:
            record_id = self._ids[index]
            record = self._records.get(record_id)

            if record is None:
                continue

            active_results.append(
                SearchResult(
                    record=record,
                    score=float(scores[index]),
                )
            )

            if len(active_results) == top_k:
                break

        return active_results