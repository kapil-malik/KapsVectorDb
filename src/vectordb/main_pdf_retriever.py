import argparse

from vectordb.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from vectordb.ingestion.chunker import RecursiveTextChunker
from vectordb.ingestion.pdf_ingestion import chunks_from_pdf
from vectordb.retrieval.semantic_text_retriever import SemanticTextRetriever
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore
from vectordb.stores.flat_nsw_inmem import FlatNSWVectorStore
from vectordb.stores.hnsw_inmem import HNSWVectorStore
from vectordb.stores.ivf_inmem import IVFVectorStore
from vectordb.stores.matrix_inmem import MatrixBackedInMemVectorStore
from vectordb.stores.naive_inmem import NaiveInMemVectorStore
from vectordb.stores.normalized_inmem import NormalizedInMemVectorStore

_STORES = {
    "naive":      lambda: NaiveInMemVectorStore(),
    "normalized": lambda: NormalizedInMemVectorStore(),
    "matrix":     lambda: MatrixBackedInMemVectorStore(),
    "buffered":   lambda: BufferedMatrixInMemVectorStore(buffer_size=1024),
    "ivf":        lambda: IVFVectorStore(nlist=50, nprobe=5, buffer_size=1024),
    "flat_nsw":   lambda: FlatNSWVectorStore(),
    "hnsw":       lambda: HNSWVectorStore(),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--store",
        choices=list(_STORES),
        default="buffered",
        help="Vector store to use (default: buffered)",
    )

    args = parser.parse_args()

    embedding_model = SentenceTransformerEmbeddingModel()
    store = _STORES[args.store]()

    retriever = SemanticTextRetriever(
        vector_store=store,
        embedding_model=embedding_model,
    )

    chunks = chunks_from_pdf(args.pdf, RecursiveTextChunker())

    retriever.add_chunks(
        texts=[chunk.text for chunk in chunks],
        metadata=[chunk.metadata for chunk in chunks],
    )

    # IVF builds the KMeans index as a separate step after all inserts.
    if isinstance(store, IVFVectorStore):
        print("Building IVF index...")
        store.build()

    results = retriever.search(args.query, top_k=args.top_k)

    print(f"Store: {args.store}")
    print(f"Indexed chunks: {len(chunks)}")
    print("\nTop retrieved chunks:")

    for result in results:
        print("\n---")
        print(f"score={result.score:.4f}")
        print(f"metadata={result.metadata}")
        print(result.text[:700])


if __name__ == "__main__":
    main()
