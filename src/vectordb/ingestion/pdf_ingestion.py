from pathlib import Path

from vectordb.ingestion.chunker import TextChunk, chunk_text
from vectordb.ingestion.pdf_loader import load_pdf_pages


def chunks_from_pdf(
        file_path: str | Path,
        chunk_size_chars: int = 800,
        overlap_chars: int = 100,
) -> list[TextChunk]:
    path = Path(file_path)
    pages = load_pdf_pages(path)

    all_chunks: list[TextChunk] = []

    for page_number, page_text in pages:
        page_chunks = chunk_text(
            text=page_text,
            chunk_size_chars=chunk_size_chars,
            overlap_chars=overlap_chars,
            base_metadata={
                "source_file": path.name,
                "page": page_number,
            },
        )

        all_chunks.extend(page_chunks)

    return all_chunks