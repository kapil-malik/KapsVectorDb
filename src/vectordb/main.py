import numpy as np

from vectordb.models import VectorRecord
from vectordb.store_base import VectorStore
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore
from vectordb.stores.file_backed import FileBackedVectorStore
from vectordb.stores.flat_nsw_inmem import FlatNSWVectorStore
from vectordb.stores.hnsw_inmem import HNSWVectorStore
from vectordb.stores.ivf_inmem import IVFVectorStore
from vectordb.stores.matrix_inmem import MatrixBackedInMemVectorStore
from vectordb.stores.mmap_store import MMapVectorStore
from vectordb.stores.naive_inmem import NaiveInMemVectorStore
from vectordb.stores.normalized_inmem import NormalizedInMemVectorStore


def insert_dummy_records(store: VectorStore) -> None:
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

def search_records(store: VectorStore) -> None:
    query_vector = np.array([1.0, 0.0, 0.0])

    results = store.search(query_vector=query_vector, top_k=2)

    print("Top results:")
    for result in results:
        print(
            f"id={result.record.id}, "
            f"score={result.score:.4f}, "
            f"text={result.record.text}"
        )

def simple_similarity_test(store: VectorStore) -> None:
    insert_dummy_records(store)
    search_records(store)


def main():
    print("Testing NaiveInMemVectorStore:")
    simple_similarity_test(NaiveInMemVectorStore())

    print("\nTesting NormalizedInMemVectorStore:")
    simple_similarity_test(NormalizedInMemVectorStore())

    print("\nTesting MatrixBackedInMemVectorStore:")
    simple_similarity_test(MatrixBackedInMemVectorStore())

    print("\nTesting BufferedMatrixInMemVectorStore:")
    simple_similarity_test(BufferedMatrixInMemVectorStore())

    print("\nTesting FileBackedVectorStore:")
    fvs = FileBackedVectorStore()
    insert_dummy_records(fvs)
    fvs.save()
    search_records(fvs)

    print("\nTesting MMapVectorStore:")
    mvs = MMapVectorStore()
    search_records(mvs)

    # IVF requires an explicit build() step after all inserts.
    # nlist=2 is used here because the dummy dataset only has 3 vectors.
    print("\nTesting IVFVectorStore:")
    ivf = IVFVectorStore(nlist=2, nprobe=1)
    insert_dummy_records(ivf)
    ivf.build()
    search_records(ivf)

    print("\nTesting FlatNSWVectorStore:")
    simple_similarity_test(FlatNSWVectorStore())

    print("\nTesting HNSWVectorStore:")
    simple_similarity_test(HNSWVectorStore())

if __name__ == "__main__":
    main()