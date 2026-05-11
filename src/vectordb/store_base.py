from typing import Protocol
import numpy as np

from vectordb.models import SearchResult, VectorRecord


class VectorStore(Protocol):
    def insert(self, record: VectorRecord) -> None:
        ...

    def get(self, record_id: str) -> VectorRecord | None:
        ...

    def delete(self, record_id: str) -> bool:
        ...

    def count(self) -> int:
        ...

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        ...