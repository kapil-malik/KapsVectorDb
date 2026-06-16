from typing import Any, Protocol
import numpy as np

from vectordb.models import SearchDiagnostics, SearchResult, VectorRecord


class VectorStore(Protocol):
    def insert(self, record: VectorRecord) -> None:
        ...

    def get(self, record_id: str) -> VectorRecord | None:
        ...

    def delete(self, record_id: str) -> bool:
        ...

    def count(self) -> int:
        ...

    def search(
            self,
            query_vector: np.ndarray,
            top_k: int = 5,
            filters: dict[str, Any] | None = None) -> list[SearchResult]:
        ...


class ANNStore(VectorStore, Protocol):
    def search_with_diagnostics(
            self,
            query_vector: np.ndarray,
            top_k: int = 5,
            filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchResult], SearchDiagnostics]:
        ...