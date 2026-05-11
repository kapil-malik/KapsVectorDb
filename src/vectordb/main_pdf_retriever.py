import argparse

from vectordb.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from vectordb.ingestion.pdf_ingestion import chunks_from_pdf
from vectordb.retrieval.semantic_text_retriever import SemanticTextRetriever
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    embedding_model = SentenceTransformerEmbeddingModel()
    vector_store = BufferedMatrixInMemVectorStore(buffer_size=1024)

    retriever = SemanticTextRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
    )

    chunks = chunks_from_pdf(args.pdf)

    retriever.add_chunks(
        texts=[chunk.text for chunk in chunks],
        metadata=[chunk.metadata for chunk in chunks],
    )

    results = retriever.search(args.query, top_k=args.top_k)

    print(f"Indexed chunks: {len(chunks)}")
    print("\nTop retrieved chunks:")

    for result in results:
        print("\n---")
        print(f"score={result.score:.4f}")
        print(f"metadata={result.metadata}")
        print(result.text[:700])


if __name__ == "__main__":
    main()