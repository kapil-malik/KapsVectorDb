import numpy as np

from vectordb.models import VectorRecord
from vectordb.store import InMemoryVectorStore


def main():
    store = InMemoryVectorStore()

    store.insert(
        VectorRecord(
            id="doc-1",
            vector=np.array([1.0, 0.0, 0.0]),
            text="I love running and fitness",
            metadata={"source": "fitness.txt"},
        )
    )

    store.insert(
        VectorRecord(
            id="doc-2",
            vector=np.array([0.0, 1.0, 0.0]),
            text="Vector databases store embeddings",
            metadata={"source": "vectordb.txt"},
        )
    )

    store.insert(
        VectorRecord(
            id="doc-3",
            vector=np.array([0.9, 0.1, 0.0]),
            text="Running improves stamina",
            metadata={"source": "running.txt"},
        )
    )

    query_vector = np.array([1.0, 0.0, 0.0])

    results = store.search(query_vector=query_vector, top_k=2)

    print("Top results:")
    for result in results:
        print(
            f"id={result.record.id}, "
            f"score={result.score:.4f}, "
            f"text={result.record.text}"
        )


if __name__ == "__main__":
    main()