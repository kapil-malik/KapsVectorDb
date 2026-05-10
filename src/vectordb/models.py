from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorRecord:
    """
    Represents one vector stored in our vector database.

    Rough Java analogy:
    - id: primary key
    - vector: float[] / double[]
    - text: original chunk text
    - metadata: Map<String, Object>
    """

    id: str
    vector: np.ndarray
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.vector.ndim != 1:
            raise ValueError("Vector must be one-dimensional")

        if self.vector.size == 0:
            raise ValueError("Vector must not be empty")


@dataclass(frozen=True)
class SearchResult:
    record: VectorRecord
    score: float