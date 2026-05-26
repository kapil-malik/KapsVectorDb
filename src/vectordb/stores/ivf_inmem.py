from typing import Any

import heapq

import numpy as np

from vectordb.models import SearchResult, VectorRecord
from vectordb.filters import metadata_matches
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore
from sklearn.cluster import MiniBatchKMeans

class IVFVectorStore(BufferedMatrixInMemVectorStore):
    """
    In-memory IVF vector store.

    IVF = Inverted File Index.

    Build process:
    - store normalized vectors
    - build KMeans centroids
    - assign each vector to nearest centroid
    - store inverted lists: centroid_id -> vector row indices

    Search process:
    - find nearest nprobe centroids
    - search only vectors in those centroid lists
    """

    def __init__(
            self,
            nlist: int = 100,
            nprobe: int = 5,
            random_state: int = 42,
            buffer_size: int = 1024):
        super().__init__(buffer_size)
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        self._nlist = nlist
        self._nprobe = nprobe
        self._random_state = random_state
        self._buffer_size = buffer_size

        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []

        self._buffer_vectors: list[np.ndarray] = []
        self._buffer_ids: list[str] = []

        self._records: dict[str, VectorRecord] = {}
        self._dim: int | None = None

        self._centroids: np.ndarray | None = None
        self._lists: dict[int, list[int]] = {}
        self._index_built = False
        self._index_stale = False


    def build(self) -> None:
        self._flush_buffer()

        if self._vectors is None or len(self._ids) == 0:
            raise ValueError("Cannot build IVF index with no vectors")

        if len(self._ids) < self._nlist:
            raise ValueError(
                f"Number of vectors ({len(self._ids)}) must be >= nlist ({self._nlist})"
            )

        kmeans = MiniBatchKMeans(
            n_clusters=self._nlist,
            random_state=self._random_state,
            batch_size=4096,
            n_init="auto",
        )

        labels = kmeans.fit_predict(self._vectors)

        centroids = kmeans.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._centroids = centroids / norms

        self._lists = {i: [] for i in range(self._nlist)}

        for row_index, cluster_id in enumerate(labels):
            self._lists[int(cluster_id)].append(row_index)

        self._index_built = True
        self._index_stale = False


    def insert(self, record: VectorRecord) -> None:
        super().insert(record)
        if self._index_built:
            self._index_stale = True


    def search(
            self,
            query_vector: np.ndarray,
            top_k: int = 5,
            filters: dict[str, Any] | None = None
        ) -> list[SearchResult]:
        if not self._index_built:
            raise ValueError("IVF index has not been built. Call build() first.")

        if self._index_stale:
            raise ValueError("IVF index is stale. Rebuild index before searching.")

        norm = np.linalg.norm(query_vector)
        if norm == 0:
            raise ValueError("Cannot search with zero vector")

        normalized_query = (query_vector / norm).astype(np.float32)

        # 1. Find nearest centroids.
        centroid_scores = self._centroids @ normalized_query

        probe_count = min(self._nprobe, len(centroid_scores))
        centroid_indices = np.argpartition(-centroid_scores, probe_count - 1)[:probe_count]

        # 2. Gather candidate row indices from selected clusters.
        candidate_indices: list[int] = []
        for centroid_id in centroid_indices:
            candidate_indices.extend(self._lists[int(centroid_id)])

        if not candidate_indices:
            return []

        candidate_matrix = self._vectors[candidate_indices]
        candidate_scores = candidate_matrix @ normalized_query

        candidate_multiplier = 10 if filters else 4
        candidate_count = min(len(candidate_scores), top_k * candidate_multiplier)

        top_candidate_positions = np.argpartition(
            -candidate_scores,
            candidate_count - 1,
            )[:candidate_count]

        candidates: list[tuple[float, str]] = []

        for position in top_candidate_positions:
            row_index = candidate_indices[position]
            record_id = self._ids[row_index]
            record = self._records.get(record_id)

            if record is None:
                continue

            if not metadata_matches(record, filters):
                continue

            candidates.append((float(candidate_scores[position]), record_id))

        top_candidates = heapq.nlargest(top_k, candidates, key=lambda item: item[0])

        return [
            SearchResult(
                record=self._records[record_id],
                score=score,
            )
            for score, record_id in top_candidates
        ]