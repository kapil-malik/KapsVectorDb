import numpy as np

from vectordb.models import VectorRecord
from vectordb.store_base import VectorStore
from vectordb.stores.naive_inmem import NaiveInMemVectorStore
from vectordb.stores.normalized_inmem import NormalizedInMemVectorStore
from vectordb.stores.matrix_inmem import MatrixBackedInMemVectorStore
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore


def simple_similarity_test(store: VectorStore) -> None:
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


def main():
    print("Testing NaiveInMemVectorStore:")
    simple_similarity_test(NaiveInMemVectorStore())

    print("\nTesting NormalizedInMemVectorStore:")
    simple_similarity_test(NormalizedInMemVectorStore())

    print("\nTesting MatrixBackedInMemVectorStore:")
    simple_similarity_test(MatrixBackedInMemVectorStore())

    print("\nTesting BufferedMatrixInMemVectorStore:")
    simple_similarity_test(BufferedMatrixInMemVectorStore())


if __name__ == "__main__":
    main()