from vectordb.embeddings.base import EmbeddingModel
from vectordb.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from vectordb.retrieval.semantic_text_retriever import SemanticTextRetriever
from vectordb.store_base import VectorStore
from vectordb.stores.buffered_matrix_inmem import BufferedMatrixInMemVectorStore
from vectordb.stores.matrix_inmem import MatrixBackedInMemVectorStore
from vectordb.stores.naive_inmem import NaiveInMemVectorStore
from vectordb.stores.normalized_inmem import NormalizedInMemVectorStore


def demo_retrieval(embedding_model: EmbeddingModel, vector_store: VectorStore):
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
        "Deadlift is a compound exercise that targets multiple muscle groups.",
        "Deadlift should be performed with proper form to avoid injury and maximize benefits.",
    ]

    retriever.add_chunks(
        texts=chunks,
        metadata=[
            {"source": "fitness"},
            {"source": "vector-db"},
            {"source": "kafka"},
            {"source": "fitness"},
            {"source": "rag"},
            {"source": "fitness"},
            {"source": "wellness"},
        ],
    )

    results = retriever.search("What is a deadlift?", top_k=3)

    print("Top retrieved chunks:")
    for result in results:
        print(f"\nscore={result.score:.4f}")
        print(f"id={result.id}")
        print(f"metadata={result.metadata}")
        print(f"text={result.text}")

    results2 = retriever.search("What is a deadlift?", top_k=3, filters={"source": "fitness"})

    print("\nTop retrieved chunks after metadata filtering:")
    for result in results2:
        print(f"\nscore={result.score:.4f}")
        print(f"id={result.id}")
        print(f"metadata={result.metadata}")
        print(f"text={result.text}")

def main():
    embedding_model = SentenceTransformerEmbeddingModel()

    print("Demonstrating retrieval with NaiveInMemVectorStore:")
    demo_retrieval(embedding_model, NaiveInMemVectorStore())

    print("\nDemonstrating retrieval with NormalizedInMemVectorStore:")
    demo_retrieval(embedding_model, NormalizedInMemVectorStore())

    print("\nDemonstrating retrieval with MatrixBackedInMemVectorStore:")
    demo_retrieval(embedding_model, MatrixBackedInMemVectorStore())

    print("\nDemonstrating retrieval with BufferedMatrixInMemVectorStore:")
    demo_retrieval(embedding_model,  BufferedMatrixInMemVectorStore())


if __name__ == "__main__":
    main()