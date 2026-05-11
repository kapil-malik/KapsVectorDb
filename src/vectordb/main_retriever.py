from vectordb.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from vectordb.retrieval.semantic_text_retriever import SemanticTextRetriever
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore


def main():
    embedding_model = SentenceTransformerEmbeddingModel()
    vector_store = BufferedMatrixInMemVectorStore(buffer_size=1024)

    retriever = SemanticTextRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
    )

    chunks = [
        "Running improves cardiovascular endurance and stamina.",
        "Vector databases store embeddings and support similarity search.",
        "Kafka is commonly used for event streaming.",
        "Strength training helps improve muscle mass and injury resilience.",
        "RAG systems retrieve relevant context before calling an LLM.",
    ]

    retriever.add_chunks(
        texts=chunks,
        metadata=[
            {"source": "fitness"},
            {"source": "vector-db"},
            {"source": "kafka"},
            {"source": "fitness"},
            {"source": "rag"},
        ],
    )

    results = retriever.search("How do embeddings help retrieve relevant documents?", top_k=3)

    print("Top retrieved chunks:")
    for result in results:
        print(f"\nscore={result.score:.4f}")
        print(f"id={result.id}")
        print(f"metadata={result.metadata}")
        print(f"text={result.text}")


if __name__ == "__main__":
    main()