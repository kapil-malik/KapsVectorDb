from pathlib import Path

from vectordb.ingestion.chunker import (
    Chunker,
    FixedSizeCharacterChunker,
    TextChunk,
)
from vectordb.ingestion.pdf_loader import load_pdf_pages


def chunks_from_pdf(
        file_path: str | Path,
        chunker: Chunker | None = None,
) -> list[TextChunk]:
    path = Path(file_path)
    pages = load_pdf_pages(path)

    actual_chunker = chunker or FixedSizeCharacterChunker()

    all_chunks: list[TextChunk] = []

    global_chunk_index = 0

    for page_number, page_text in pages:
        page_chunks = actual_chunker.chunk(
            text=page_text,
            base_metadata={
                "source_file": path.name,
                "page": page_number,
            },
        )

        for chunk in page_chunks:
            metadata = dict(chunk.metadata)
            metadata["global_chunk_index"] = global_chunk_index
            global_chunk_index += 1

            all_chunks.append(
                TextChunk(
                    text=chunk.text,
                    metadata=metadata,
                )
            )

    return all_chunks