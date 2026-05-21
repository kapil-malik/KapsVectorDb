from vectordb.ingestion.chunker import (
    FixedSizeCharacterChunker,
    RecursiveTextChunker,
)


SAMPLE_TEXT = """
Vector databases are specialized systems for storing and searching high-dimensional vectors.
They are commonly used in semantic search, recommendation systems, and RAG applications.

Chunking is an important part of retrieval augmented generation. If chunks are too small,
they may miss important context. If chunks are too large, their embeddings may become noisy
because the chunk may contain multiple unrelated ideas.

A good chunking strategy tries to preserve semantic coherence. Paragraphs, sections, headings,
and sentence boundaries can all help produce better chunks for retrieval.
"""


def print_chunks(name: str, chunks):
    print(f"\n{name}")
    print("=" * len(name))

    for chunk in chunks:
        print("\n---")
        print(chunk.metadata)
        print(chunk.text)


def main():
    fixed = FixedSizeCharacterChunker(
        chunk_size_chars=200,
        overlap_chars=40,
    )

    recursive = RecursiveTextChunker(
        chunk_size_chars=200,
        overlap_chars=40,
    )

    fixed_chunks = fixed.chunk(SAMPLE_TEXT)
    recursive_chunks = recursive.chunk(SAMPLE_TEXT)

    print_chunks("FixedSizeCharacterChunker", fixed_chunks)
    print_chunks("RecursiveTextChunker", recursive_chunks)


if __name__ == "__main__":
    main()