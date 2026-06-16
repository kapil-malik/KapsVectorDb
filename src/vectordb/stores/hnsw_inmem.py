from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Any

import numpy as np

from vectordb.distance import dot_similarity
from vectordb.filters import metadata_matches
from vectordb.models import SearchDiagnostics, SearchResult, VectorRecord


class HNSWVectorStore:
    """
    Simple in-memory HNSW vector store.

    Learning goals:
    - hierarchical graph-based ANN
    - greedy upper-layer traversal
    - ef_construction vs ef_search
    - recall/latency tradeoffs
    """

    def __init__(
            self,
            m: int = 8,
            ef_construction: int = 64,
            ef_search: int = 32,
            level_multiplier: float = 1.0,
    ):
        if m <= 0:
            raise ValueError("m must be > 0")
        if ef_construction <= 0:
            raise ValueError("ef_construction must be > 0")
        if ef_search <= 0:
            raise ValueError("ef_search must be > 0")

        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.level_multiplier = level_multiplier

        self._records: dict[str, VectorRecord] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._levels: dict[str, int] = {}

        # level -> node_id -> set(neighbor_ids)
        self._neighbors: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

        self._tombstone_ids: set[str] = set()

        self._entry_point_id: str | None = None
        self._max_level: int = -1

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

        new_level = self._random_level()
        self._levels[record.id] = new_level

        if self._entry_point_id is None:
            self._entry_point_id = record.id
            self._max_level = new_level
            return

        current_entry = self._entry_point_id

        # 1. Greedy search from top layer down to just above new node level.
        for level in range(self._max_level, new_level, -1):
            current_entry = self._greedy_search_layer(normalized_vector, current_entry, level)

        # 2. Search and connect on all layers where the new node participates.
        max_connect_level = min(new_level, self._max_level)

        for level in range(max_connect_level, -1, -1):
            candidates = self._search_layer(
                query_vector=normalized_vector,
                entry_id=current_entry,
                level=level,
                ef=self.ef_construction,
            )

            neighbor_ids = self._select_neighbors(candidates, max_neighbors=self.m, exclude_id=record.id)

            for neighbor_id in neighbor_ids:
                self._neighbors[level][record.id].add(neighbor_id)
                self._neighbors[level][neighbor_id].add(record.id)
                self._prune_neighbors(level, neighbor_id)

            if candidates:
                current_entry = max(candidates, key=lambda item: item[0])[1]

        # 3. If this node is now the highest-level node, make it the entry point.
        if new_level > self._max_level:
            self._entry_point_id = record.id
            self._max_level = new_level

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
            self._max_level = -1

            for candidate_id, level in self._levels.items():
                if candidate_id != record_id and candidate_id not in self._tombstone_ids:
                    if level > self._max_level:
                        self._max_level = level
                        self._entry_point_id = candidate_id

        return True

    def count(self) -> int:
        return len(self._records) - len(self._tombstone_ids)

    def search(
            self,
            query_vector: np.ndarray,
            top_k: int = 5,
            filters: dict[str, Any] | None = None) -> list[SearchResult]:
        results, _ = self.search_with_diagnostics(query_vector, top_k, filters)
        return results

    def search_with_diagnostics(
            self,
            query_vector: np.ndarray,
            top_k: int = 5,
            filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchResult], SearchDiagnostics]:
        diag = SearchDiagnostics()

        if self._entry_point_id is None:
            return [], diag

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

        current_entry = self._entry_point_id

        # 1. Greedy descent from top layer to layer 1.
        for level in range(self._max_level, 0, -1):
            current_entry = self._greedy_search_layer(normalized_query_vector, current_entry, level, diag)

        # 2. Wider ef_search at layer 0.
        best_candidates = self._search_layer(
            query_vector=normalized_query_vector,
            entry_id=current_entry,
            level=0,
            ef=self.ef_search,
            diag=diag,
        )

        diag.layers_traversed = self._max_level + 1

        best_candidates.sort(key=lambda x: x[0], reverse=True)

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

        return results, diag


    # HNSW uses a random level generator based on a geometric distribution.
    def _random_level(self) -> int:
        u = np.random.random()
        return int(-np.log(u) * self.level_multiplier)


    # Greedy search for upper layers.
    def _greedy_search_layer(
            self,
            query_vector: np.ndarray,
            entry_id: str,
            level: int,
            diag: SearchDiagnostics | None = None,
        ) -> str:
        current_id = entry_id
        current_score = dot_similarity(query_vector, self._vectors[current_id])
        if diag:
            diag.distance_computations += 1
            if diag.visited_node_ids is not None:
                diag.visited_node_ids.append(entry_id)

        improved = True

        while improved:
            improved = False
            for neighbor_id in self._neighbors[level][current_id]:
                if diag: diag.graph_hops += 1
                if neighbor_id in self._tombstone_ids:
                    continue

                neighbor_score = dot_similarity(query_vector, self._vectors[neighbor_id])
                if diag:
                    diag.distance_computations += 1
                    if diag.visited_node_ids is not None:
                        diag.visited_node_ids.append(neighbor_id)
                if neighbor_score > current_score:
                    current_id = neighbor_id
                    current_score = neighbor_score
                    improved = True

        return current_id


    # Search for neighbors starting from the entry point, but only within the specified layer.
    def _search_layer(
            self,
            query_vector: np.ndarray,
            entry_id: str,
            level: int,
            ef: int,
            diag: SearchDiagnostics | None = None,
    ) -> list[tuple[float, str]]:
        visited: set[str] = set()

        # Min heap by negative score, so best score comes first.
        candidates: list[tuple[float, str]] = []

        # Min heap of best results found so far.
        # Stores (score, id), smallest score at root.
        best_results: list[tuple[float, str]] = []

        entry_score = dot_similarity(query_vector, self._vectors[entry_id])
        if diag: diag.distance_computations += 1

        heapq.heappush(candidates, (-entry_score, entry_id))
        heapq.heappush(best_results, (entry_score, entry_id))
        visited.add(entry_id)
        if diag:
            diag.visited_nodes += 1
            if diag.visited_node_ids is not None:
                diag.visited_node_ids.append(entry_id)

        while candidates:
            current_neg_score, current_id = heapq.heappop(candidates)
            current_score = -current_neg_score

            worst_best_score = best_results[0][0]

            if len(best_results) >= ef and current_score < worst_best_score:
                break

            for neighbor_id in self._neighbors[level].get(current_id, set()):
                if diag: diag.graph_hops += 1
                if neighbor_id in visited or neighbor_id in self._tombstone_ids:
                    continue

                visited.add(neighbor_id)
                if diag:
                    diag.visited_nodes += 1
                    if diag.visited_node_ids is not None:
                        diag.visited_node_ids.append(neighbor_id)

                neighbor_score = dot_similarity(query_vector, self._vectors[neighbor_id])
                if diag: diag.distance_computations += 1

                if len(best_results) < ef or neighbor_score > worst_best_score:
                    heapq.heappush(candidates, (-neighbor_score, neighbor_id))
                    heapq.heappush(best_results, (neighbor_score, neighbor_id))

                    if len(best_results) > ef:
                        heapq.heappop(best_results)

        return best_results


    # After collecting candidate neighbors,
    # we need to select the top M to connect to, excluding any tombstoned records.
    def _select_neighbors(
            self,
            candidates: list[tuple[float, str]],
            max_neighbors: int,
            exclude_id: str | None = None,
    ) -> list[str]:
        filtered = [
            (score, record_id)
            for score, record_id in candidates
            if record_id != exclude_id and record_id not in self._tombstone_ids
        ]

        top = heapq.nlargest(max_neighbors, filtered, key=lambda item: item[0])
        return [record_id for _, record_id in top]


    # Prune if the record_id has more than M neighbors, keep only the top M most similar neighbors.
    def _prune_neighbors(self, level: int, record_id: str):
        neighbors = self._neighbors[level].get(record_id, set())
        if len(neighbors) <= self.m:
            return

        record_vector = self._vectors[record_id]
        neighbor_scores = {
            nbr_id: dot_similarity(record_vector, self._vectors[nbr_id])
            for nbr_id in neighbors
            if nbr_id not in self._tombstone_ids
        }

        # Get the M neighbors with the highest similarity scores
        top_neighbors = heapq.nlargest(self.m, neighbor_scores.items(), key=lambda x: x[1])
        top_neighbor_ids = set(nbr_id for nbr_id, _ in top_neighbors)

        self._neighbors[level][record_id] = top_neighbor_ids
