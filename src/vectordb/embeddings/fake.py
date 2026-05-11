import hashlib

import numpy as np


class FakeHashEmbeddingModel:
    """
    Deterministic fake embedding model.

    Useful for testing plumbing without downloading a real model.

    Important:
    - This does NOT produce semantically meaningful embeddings.
    - Same input text always produces same vector.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()

        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)

        return rng.random(self._dimension).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(text) for text in texts]