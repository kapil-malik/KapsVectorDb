import heapq
from typing import Any

import numpy as np

from vectordb.filters import metadata_matches
from vectordb.models import SearchResult, VectorRecord


class MMapVectorStore:
    """
    Read-optimized vector store backed by a memory-mapped NumPy matrix.

    This store loads:
    - metadata from records.jsonl
    - vectors from vectors.npy using mmap_mode="r"
    - tombstones from tombstones.txt

    It is designed to search persisted vector files without eagerly loading the
    full vector matrix into RAM.
    """

    def __init__(
            self,
            records_file: str = "records.jsonl",
            vectors_file: str = "vectors.npy",
            tombstones_file: str = "tombstones.txt",
            buffer_size: int = 1024):
        self._records_file = records_file
        self._vectors_file = vectors_file
        self._tombstones_file = tombstones_file

        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self._buffer_size = buffer_size

        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []

        self._buffer_vectors: list[np.ndarray] = []
        self._buffer_ids: list[str] = []

        self._records: dict[str, VectorRecord] = {}
        self._tombstones = set()
        self._dim: int | None = None
        self.load()

    def load(self):
        try:
            with open(self._tombstones_file, "r") as f:
                self._tombstones = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            self._tombstones = set()

        try:
            self._vectors = np.load(self._vectors_file, mmap_mode="r")
            vector_count = self._vectors.shape[0]

            with open(self._records_file, "r") as f:
                for line in f:
                    v_index = len(self._ids)
                    if v_index >= vector_count:
                        raise ValueError("records.jsonl has more records than vectors.npy")

                    vector = self._vectors[v_index]
                    record = VectorRecord.from_metadata_json(line.strip(), vector)

                    self._ids.append(record.id)

                    if record.id not in self._tombstones:
                        self._records[record.id] = record

                    if self._dim is None:
                        self._dim = vector.size

        except FileNotFoundError:
            self._vectors = None

    def save(self):
        if self._buffer_ids:
            with open(self._records_file, "a") as f:
                for record_id in self._buffer_ids:
                    record = self._records[record_id]
                    f.write(record.to_metadata_json() + "\n")

            batch_matrix = np.vstack(self._buffer_vectors).astype(np.float32)

            if self._vectors is None:
                self._vectors = batch_matrix
            else:
                self._vectors = np.vstack([self._vectors, batch_matrix])

            self._ids.extend(self._buffer_ids)

            np.save(self._vectors_file, self._vectors)

            self._buffer_vectors.clear()
            self._buffer_ids.clear()

        self._save_tombstones()

    def insert(self, record: VectorRecord) -> None:
        """
        MMapVectorStore is read-optimized.

        For now, use BufferedMatrixFileVectorStore for writes, then reload this
        store for mmap-based reads.
        """
        raise NotImplementedError(
            "MMapVectorStore is read-only for now. "
            "Use FileBackedVectorStore for inserts, then reload MMapVectorStore."
        )

    def get(self, record_id: str) -> VectorRecord | None:
        return self._get_live_record(record_id)

    def delete(self, record_id: str) -> bool:
        if not self._is_live_record(record_id):
            return False

        del self._records[record_id]
        self._tombstones.add(record_id)
        self._append_tombstone(record_id)
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

                record = self._get_live_record(record_id)
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

                record = self._get_live_record(record_id)
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

    def compact(self) -> None:
        """
        Physically removes tombstoned/deleted records from disk and memory.

        After compaction:
        - records.jsonl contains only live records
        - vectors.npy contains only live vectors
        - tombstones.txt is cleared
        - in-memory matrix/id mappings are rebuilt
        """
        # First flush pending inserts so everything is represented in main files/matrix.
        self.save()

        live_ids: list[str] = []
        live_vectors: list[np.ndarray] = []
        live_records: dict[str, VectorRecord] = {}

        # _ids preserves row order for _vectors.
        for index, record_id in enumerate(self._ids):
            record = self._get_live_record(record_id)
            if record is None:
                continue

            live_ids.append(record_id)
            live_vectors.append(self._vectors[index])
            live_records[record_id] = record

        if live_vectors:
            new_vectors = np.vstack(live_vectors).astype(np.float32)
        else:
            new_vectors = None

        # Rewrite records file.
        with open(self._records_file, "w") as f:
            for record_id in live_ids:
                record = live_records[record_id]
                f.write(record.to_metadata_json() + "\n")

        # Rewrite vectors file.
        if new_vectors is not None:
            np.save(self._vectors_file, new_vectors)
        else:
            # For simplicity, store an empty matrix if dimension is known.
            if self._dim is not None:
                empty = np.empty((0, self._dim), dtype=np.float32)
                np.save(self._vectors_file, empty)
                new_vectors = empty
            else:
                self._vectors = None

        # Clear tombstones file.
        with open(self._tombstones_file, "w") as f:
            pass

        # Rebuild in-memory state.
        self._ids = live_ids
        self._vectors = new_vectors
        self._records = live_records
        self._tombstones.clear()
        self._buffer_ids.clear()
        self._buffer_vectors.clear()

    def _save_tombstones(self) -> None:
        with open(self._tombstones_file, "w") as f:
            for tombstone in sorted(self._tombstones):
                f.write(tombstone + "\n")

    def _append_tombstone(self, tombstone: str) -> None:
        with open(self._tombstones_file, "a") as f:
            f.write(tombstone + "\n")

    def _is_live_record(self, record_id: str) -> bool:
        return record_id in self._records and record_id not in self._tombstones

    def _get_live_record(self, record_id: str) -> VectorRecord:
        return self._records[record_id] if self._is_live_record(record_id) else None