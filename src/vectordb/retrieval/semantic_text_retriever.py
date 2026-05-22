from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from vectordb.embeddings.base import EmbeddingModel
from vectordb.models import VectorRecord
from vectordb.store_base import VectorStore


@dataclass(frozen=True)
class RetrievedTextChunk:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticTextRetriever:
    """
    User-facing abstraction for semantic text retrieval.

    External contract:
    - add text chunks
    - search by natural language text
    - return matching text chunks

    Internal implementation:
    - embed text
    - store vector records
    - run vector search
    """

    def __init__(
            self,
            vector_store: VectorStore,
            embedding_model: EmbeddingModel,
    ):
        self._vector_store = vector_store
        self._embedding_model = embedding_model

    def add_chunk(
            self,
            text: str,
            chunk_id: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> str:
        if not text.strip():
            raise ValueError("text must not be empty")

        actual_chunk_id = chunk_id or str(uuid4())
        vector = self._embedding_model.embed(text)

        record = VectorRecord(
            id=actual_chunk_id,
            vector=vector,
            text=text,
            metadata=metadata or {},
        )

        self._vector_store.insert(record)

        return actual_chunk_id

    def add_chunks(
            self,
            texts: list[str],
            metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("metadata length must match texts length")

        vectors = self._embedding_model.embed_batch(texts)

        chunk_ids: list[str] = []

        for i, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"text at index {i} must not be empty")

            chunk_id = str(uuid4())
            chunk_ids.append(chunk_id)

            record = VectorRecord(
                id=chunk_id,
                vector=vectors[i],
                text=text,
                metadata=metadata[i] if metadata is not None else {},
            )

            self._vector_store.insert(record)

        return chunk_ids

    def search(
            self,
            query: str,
            top_k: int = 5,
            filters: dict[str, Any] | None = None,
    ) -> list[RetrievedTextChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")

        query_vector = self._embedding_model.embed(query)
        results = self._vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters)

        return [
            RetrievedTextChunk(
                id=result.record.id,
                text=result.record.text,
                score=result.score,
                metadata=result.record.metadata,
            )
            for result in results
        ]