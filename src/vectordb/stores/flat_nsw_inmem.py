from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Any

import numpy as np

from vectordb.distance import dot_similarity
from vectordb.filters import metadata_matches
from vectordb.models import SearchResult, VectorRecord


class FlatNSWVectorStore:
    """
    Flat Navigable Small World graph.

    This is NOT full HNSW yet.
    It is a single-layer graph ANN index.

    Learning goals:
    - graph-based approximate search
    - ef_search tradeoff
    - neighbor connectivity
    - recall vs latency comparison
    """

    def __init__(
            self,
            m: int = 8,
            ef_search: int = 32,
    ):
        if m <= 0:
            raise ValueError("m must be > 0")

        if ef_search <= 0:
            raise ValueError("ef_search must be > 0")

        self.m = m
        self.ef_search = ef_search

        self._records: dict[str, VectorRecord] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._neighbors: dict[str, set[str]] = defaultdict(set)
        self._tombstone_ids: set[str] = set()

        self._entry_point_id: str | None = None

    def insert(self, record: VectorRecord) -> None:
        if record.id in self._records:
            raise ValueError(f"Record with id '{record.id}' already exists")

        if record.vector.ndim != 1:
            raise ValueError("Vector must be one-dimensional")

        if record.vector.size == 0:
            raise ValueError("Vector must not be empty")

        norm = np.linalg.norm(record.vector)
        if norm == 0:
            raise ValueError("Cannot insert zero vector")

        normalized_vector = (record.vector / norm).astype(np.float32)

        self._records[record.id] = record
        self._vectors[record.id] = normalized_vector

        if self._entry_point_id is None:
            self._entry_point_id = record.id
            return

        # Connect to M nearest neighbors
        neighbors = self._search_neighbors(normalized_vector, self.m, exclude_id=record.id)

        for neighbor_id in neighbors:
            self._neighbors[record.id].add(neighbor_id)
            self._neighbors[neighbor_id].add(record.id)
            self._prune_neighbors(neighbor_id)

    def get(self, record_id: str) -> VectorRecord | None:
        if record_id in self._tombstone_ids:
            return None

        return self._records.get(record_id)

    def delete(self, record_id: str) -> bool:
        if record_id not in self._records or record_id in self._tombstone_ids:
            return False

        self._tombstone_ids.add(record_id)

        if self._entry_point_id == record_id:
            self._entry_point_id = None
            for candidate_id in self._records:
                if candidate_id != record_id and candidate_id not in self._tombstone_ids:
                    self._entry_point_id = candidate_id
                    break

        return True

    def count(self) -> int:
        return len(self._records) - len(self._tombstone_ids)

    def search(
            self,
            query_vector: np.ndarray,
            top_k: int = 5,
            filters: dict[str, Any] | None = None) -> list[SearchResult]:
        if self._entry_point_id is None:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if query_vector.ndim != 1:
            raise ValueError("Query vector must be one-dimensional")

        if query_vector.size == 0:
            raise ValueError("Query vector must not be empty")

        norm = np.linalg.norm(query_vector)
        if norm == 0:
            raise ValueError("Cannot search with zero vector")

        normalized_query_vector = (query_vector / norm).astype(np.float32)

        # Perform search using the entry point
        best_candidates = self._search_with_entry_point(normalized_query_vector)
        best_candidates.sort(key=lambda x: x[0], reverse=True)  # Sort by score descending

        results: list[SearchResult] = []

        for score, record_id in best_candidates:
            if record_id in self._tombstone_ids:
                continue

            record = self._records[record_id]
            if record is None:
                continue

            if not metadata_matches(record, filters):
                continue

            results.append(SearchResult(record=record, score=score))

            if len(results) >= top_k:
                break

        return results


    # Prune if the record_id has more than M neighbors, keep only the top M most similar neighbors.
    def _prune_neighbors(self, record_id: str):
        if len(self._neighbors[record_id]) <= self.m:
            return

        record_vector = self._vectors[record_id]
        neighbor_scores = {
            nbr_id: dot_similarity(record_vector, self._vectors[nbr_id])
            for nbr_id in self._neighbors[record_id]
        }

        # Get the M neighbors with the highest similarity scores
        top_neighbors = heapq.nlargest(self.m, neighbor_scores.items(), key=lambda x: x[1])
        top_neighbor_ids = set(nbr_id for nbr_id, _ in top_neighbors)

        self._neighbors[record_id] = top_neighbor_ids


    def _search_neighbors(
        self,
        normalized_vector: np.ndarray,
        neighbor_count: int,
        exclude_id: str | None = None) -> list[str]:
        scores = {
            record_id: dot_similarity(normalized_vector, vec)
            for record_id, vec in self._vectors.items()
            if record_id not in self._tombstone_ids
        }

        top_neighbors = heapq.nlargest(neighbor_count, scores.items(), key=lambda x: x[1])
        return [record_id for record_id, _ in top_neighbors]


    def _search_with_entry_point(self, normalized_query_vector: np.ndarray) -> list[tuple[float, str]]:
        visited = set()

        # Min heap by negative score, so best score comes first.
        candidates: list[tuple[float, str]] = []

        # Min heap of best results found so far.
        # Stores (score, id), smallest score at root.
        best_results: list[tuple[float, str]] = []

        entry_id = self._entry_point_id
        entry_score = dot_similarity(normalized_query_vector, self._vectors[entry_id])

        heapq.heappush(candidates, (-entry_score, entry_id))
        heapq.heappush(best_results, (entry_score, entry_id))
        visited.add(entry_id)

        while candidates:
            current_score, current_id = heapq.heappop(candidates)
            current_score = -current_score

            lowest_score = best_results[0][0] if best_results else float('-inf')

            # If we already have enough results and
            # the best candidate is no better than the worst result we have, we can stop.
            if len(best_results) >= self.ef_search and current_score < lowest_score:
                break

            for neighbor_id in self._neighbors[current_id]:
                if neighbor_id in visited or neighbor_id in self._tombstone_ids:
                    continue

                visited.add(neighbor_id)

                neighbor_vector = self._vectors[neighbor_id]
                neighbor_score = dot_similarity(normalized_query_vector, neighbor_vector)

                # If we have room in results, add it.
                if len(best_results) < self.ef_search or neighbor_score > lowest_score:
                    heapq.heappush(candidates, (-neighbor_score, neighbor_id))
                    heapq.heappush(best_results, (neighbor_score, neighbor_id))

                    if len(best_results) > self.ef_search:
                        # Remove worst result to maintain size
                        heapq.heappop(best_results)

        return best_results

