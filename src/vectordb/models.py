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


@dataclass
class SearchDiagnostics:
    distance_computations: int = 0  # dot_similarity calls — universal cost proxy
    visited_nodes: int = 0          # unique nodes added to visited set (graph stores)
    graph_hops: int = 0             # total neighbor-list iterations (graph stores)
    layers_traversed: int = 0       # layers descended through (HNSW: max_level+1, NSW: 1)
    clusters_scanned: int = 0       # nprobe clusters probed (IVF only)
    vectors_scanned: int = 0        # candidate vectors in selected clusters (IVF only)
    visited_node_ids: list[str] | None = None  # opt-in: IDs added to visited set; None = don't collect