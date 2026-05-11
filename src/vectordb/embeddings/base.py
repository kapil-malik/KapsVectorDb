from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    """
    Converts text into dense vectors.

    This is intentionally pluggable.
    The vector DB does not care whether embeddings come from:
    - a fake test model
    - sentence-transformers
    - OpenAI
    - Cohere
    - BGE
    """

    @property
    def dimension(self) -> int:
        ...

    def embed(self, text: str) -> np.ndarray:
        ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        ...