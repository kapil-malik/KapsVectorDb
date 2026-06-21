from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Any

import numpy as np

from vectordb.distance import dot_similarity
from vectordb.filters import metadata_matches
from vectordb.models import SearchDiagnostics, SearchResult, VectorRecord


class FlatNSWV2VectorStore:
    """
    Flat Navigable Small World graph — approximate construction variant.

    Differs from FlatNSWVectorStore in how the graph is built:
    - v1 insert: exact scan over all vectors to pick M nearest neighbors  → O(N^2) build
    - v2 insert: graph-based beam search with ef_construction candidates  → sub-quadratic build

    Learning goals:
    - how ef_construction trades build speed against graph quality
    - approximate vs exact neighbor selection during index construction
    - comparing recall/latency between v1 and v2 at similar M / ef_search
    """

    def __init__(
            self,
            m: int = 8,
            ef_search: int = 32,
            ef_construction: int = 64,
    ):
        if m <= 0:
            raise ValueError("m must be > 0")

        if ef_search <= 0:
            raise ValueError("ef_search must be > 0")

        if ef_construction <= 0:
            raise ValueError("ef_construction must be > 0")

        self.m = m
        self.ef_search = ef_search
        self.ef_construction = ef_construction

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

        if self._entry_point_id is None:
            self._records[record.id] = record
            self._vectors[record.id] = normalized_vector
            self._entry_point_id = record.id
            return

        # Search the existing graph before adding the new record so it cannot
        # appear as its own neighbor candidate.
        candidates = self._search_with_entry_point(normalized_vector, ef=self.ef_construction)
        neighbor_ids = self._select_neighbors(candidates, self.m)

        self._records[record.id] = record
        self._vectors[record.id] = normalized_vector

        for neighbor_id in neighbor_ids:
            self._neighbors[record.id].add(neighbor_id)
            self._neighbors[neighbor_id].add(record.id)
            self._prune_neighbors(neighbor_id)

        self._prune_neighbors(record.id)

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
        results, _ = self.search_with_diagnostics(query_vector, top_k, filters)
        return results

    def search_with_diagnostics(
            self,
            query_vector: np.ndarray,
            top_k: int = 5,
            filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchResult], SearchDiagnostics]:
        diag = SearchDiagnostics(layers_traversed=1)

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

        best_candidates = self._search_with_entry_point(normalized_query_vector, diag=diag)
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


    # Prune if record_id has more than M neighbors; keep only the top M by similarity.
    def _prune_neighbors(self, record_id: str):
        if len(self._neighbors[record_id]) <= self.m:
            return

        record_vector = self._vectors[record_id]
        neighbor_scores = {
            nbr_id: dot_similarity(record_vector, self._vectors[nbr_id])
            for nbr_id in self._neighbors[record_id]
        }

        top_neighbors = heapq.nlargest(self.m, neighbor_scores.items(), key=lambda x: x[1])
        self._neighbors[record_id] = set(nbr_id for nbr_id, _ in top_neighbors)


    def _select_neighbors(
            self,
            candidates: list[tuple[float, str]],
            max_neighbors: int,
            exclude_id: str | None = None,
    ) -> list[str]:
        filtered = [(score, rid) for score, rid in candidates if rid != exclude_id]
        top = heapq.nlargest(max_neighbors, filtered, key=lambda x: x[0])
        return [rid for _, rid in top]


    def _search_with_entry_point(
            self,
            normalized_query_vector: np.ndarray,
            ef: int | None = None,
            diag: SearchDiagnostics | None = None,
    ) -> list[tuple[float, str]]:
        ef_limit = ef if ef is not None else self.ef_search

        visited = set()

        # Min heap by negative score, so best score comes first.
        candidates: list[tuple[float, str]] = []

        # Min heap of best results found so far.
        # Stores (score, id), smallest score at root.
        best_results: list[tuple[float, str]] = []

        entry_id = self._entry_point_id
        entry_score = dot_similarity(normalized_query_vector, self._vectors[entry_id])
        if diag: diag.distance_computations += 1

        heapq.heappush(candidates, (-entry_score, entry_id))
        heapq.heappush(best_results, (entry_score, entry_id))
        visited.add(entry_id)
        if diag:
            diag.visited_nodes += 1
            if diag.visited_node_ids is not None:
                diag.visited_node_ids.append(entry_id)

        while candidates:
            current_score, current_id = heapq.heappop(candidates)
            current_score = -current_score

            lowest_score = best_results[0][0] if best_results else float('-inf')

            # If we already have enough results and the best remaining candidate
            # cannot improve the worst result we hold, stop.
            if len(best_results) >= ef_limit and current_score < lowest_score:
                break

            for neighbor_id in self._neighbors[current_id]:
                if diag: diag.graph_hops += 1
                if neighbor_id in visited or neighbor_id in self._tombstone_ids:
                    continue

                visited.add(neighbor_id)
                if diag:
                    diag.visited_nodes += 1
                    if diag.visited_node_ids is not None:
                        diag.visited_node_ids.append(neighbor_id)

                neighbor_score = dot_similarity(normalized_query_vector, self._vectors[neighbor_id])
                if diag: diag.distance_computations += 1

                if len(best_results) < ef_limit or neighbor_score > lowest_score:
                    heapq.heappush(candidates, (-neighbor_score, neighbor_id))
                    heapq.heappush(best_results, (neighbor_score, neighbor_id))

                    if len(best_results) > ef_limit:
                        heapq.heappop(best_results)

        return best_results
