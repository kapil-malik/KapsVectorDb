from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    metadata: dict


def chunk_text(
        text: str,
        chunk_size_chars: int = 800,
        overlap_chars: int = 100,
        base_metadata: dict | None = None,
) -> list[TextChunk]:
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be positive")

    if overlap_chars < 0:
        raise ValueError("overlap_chars must not be negative")

    if overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be smaller than chunk_size_chars")

    cleaned = " ".join(text.split())

    if not cleaned:
        return []

    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0

    while start < len(cleaned):
        end = min(start + chunk_size_chars, len(cleaned))
        chunk = cleaned[start:end]

        metadata = dict(base_metadata or {})
        metadata["chunk_index"] = chunk_index
        metadata["start_char"] = start
        metadata["end_char"] = end

        chunks.append(TextChunk(text=chunk, metadata=metadata))

        if end == len(cleaned):
            break

        start = end - overlap_chars
        chunk_index += 1

    return chunks