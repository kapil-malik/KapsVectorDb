from dataclasses import dataclass, field
from typing import Any

import numpy as np
import json


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

    @classmethod
    def from_metadata_json(cls, line: str, vector: np.ndarray | None = None) -> "VectorRecord":
        data = json.loads(line)
        return cls(
            id=data["id"],
            vector=vector,
            text=data["text"],
            metadata=data.get("metadata", {}),
        )

    def to_metadata_json(self):
        return json.dumps(
            {
                "id": self.id,
                "text": self.text,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class SearchResult:
    record: VectorRecord
    score: float