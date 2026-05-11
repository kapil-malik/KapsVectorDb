import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingModel:
    """
    Embedding model backed by sentence-transformers.

    Recommended starter model:
    sentence-transformers/all-MiniLM-L6-v2

    Dimension: 384
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [vector.astype(np.float32) for vector in vectors]