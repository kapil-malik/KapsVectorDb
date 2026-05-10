import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity measures the angle between two vectors.

    Result range:
    - 1.0  means same direction
    - 0.0  means unrelated / orthogonal
    - -1.0 means opposite direction
    """

    if a.shape != b.shape:
        raise ValueError(f"Vector shapes must match. Got {a.shape} and {b.shape}")

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")

    return float(dot_product / (norm_a * norm_b))