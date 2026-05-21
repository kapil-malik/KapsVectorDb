from dataclasses import dataclass, field
from typing import Protocol, Any


@dataclass(frozen=True)
class TextChunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker(Protocol):
    def chunk(
            self,
            text: str,
            base_metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        ...

class FixedSizeCharacterChunker:
    """
    Simple chunker.

    It ignores paragraph/sentence boundaries and chunks by character count.
    Good baseline, but can split in awkward places.
    """

    def __init__(
            self,
            chunk_size_chars: int = 800,
            overlap_chars: int = 100,
    ):
        if chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars must be positive")

        if overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative")

        if overlap_chars >= chunk_size_chars:
            raise ValueError("overlap_chars must be smaller than chunk_size_chars")

        self._chunk_size_chars = chunk_size_chars
        self._overlap_chars = overlap_chars

    def chunk(
            self,
            text: str,
            base_metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        cleaned = " ".join(text.split())

        if not cleaned:
            return []

        chunks: list[TextChunk] = []
        start = 0
        chunk_index = 0

        while start < len(cleaned):
            end = min(start + self._chunk_size_chars, len(cleaned))
            chunk_text = cleaned[start:end]

            metadata = dict(base_metadata or {})
            metadata["chunk_index"] = chunk_index
            metadata["start_char"] = start
            metadata["end_char"] = end
            metadata["chunker"] = "fixed_size_character"

            chunks.append(TextChunk(text=chunk_text, metadata=metadata))

            if end == len(cleaned):
                break

            start = end - self._overlap_chars
            chunk_index += 1

        return chunks


class RecursiveTextChunker:
    """
    Tries to preserve natural text boundaries.

    Strategy:
    1. Try splitting by paragraphs.
    2. If too large, split by sentences.
    3. If still too large, split by words.
    4. If still too large, fall back to characters.

    This is much better for RAG than raw fixed-size character chunking.
    """

    def __init__(
            self,
            chunk_size_chars: int = 800,
            overlap_chars: int = 100,
    ):
        if chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars must be positive")

        if overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative")

        if overlap_chars >= chunk_size_chars:
            raise ValueError("overlap_chars must be smaller than chunk_size_chars")

        self._chunk_size_chars = chunk_size_chars
        self._overlap_chars = overlap_chars

    def chunk(
            self,
            text: str,
            base_metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        cleaned = text.strip()

        if not cleaned:
            return []

        pieces = self._split_recursively(cleaned)
        merged = self._merge_pieces(pieces)

        chunks: list[TextChunk] = []

        for i, chunk_text in enumerate(merged):
            metadata = dict(base_metadata or {})
            metadata["chunk_index"] = i
            metadata["chunker"] = "recursive_text"
            metadata["length_chars"] = len(chunk_text)

            chunks.append(TextChunk(text=chunk_text, metadata=metadata))

        return chunks

    def _split_recursively(self, text: str) -> list[str]:
        if len(text) <= self._chunk_size_chars:
            return [self._normalize_spaces(text)]

        for separator in ["\n\n", ". ", "? ", "! ", "\n", " "]:
            if separator in text:
                pieces = text.split(separator)

                result: list[str] = []
                for i, piece in enumerate(pieces):
                    piece = piece.strip()
                    if not piece:
                        continue

                    # Add sentence punctuation back where needed.
                    if separator in [". ", "? ", "! "] and i < len(pieces) - 1:
                        piece = piece + separator.strip()

                    result.extend(self._split_recursively(piece))

                return result

        # Last fallback: hard character split.
        return [
            text[i : i + self._chunk_size_chars]
            for i in range(0, len(text), self._chunk_size_chars)
        ]

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for piece in pieces:
            piece = self._normalize_spaces(piece)

            if not piece:
                continue

            additional_len = len(piece) + (1 if current_parts else 0)

            if current_len + additional_len <= self._chunk_size_chars:
                current_parts.append(piece)
                current_len += additional_len
            else:
                if current_parts:
                    chunks.append(" ".join(current_parts))

                current_parts = [piece]
                current_len = len(piece)

        if current_parts:
            chunks.append(" ".join(current_parts))

        if self._overlap_chars > 0:
            chunks = self._add_overlap(chunks)

        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        if len(chunks) <= 1:
            return chunks

        overlapped: list[str] = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
                continue

            previous = chunks[i - 1]
            overlap = previous[-self._overlap_chars:]

            # Find the first space to skip any broken word at the start
            first_space_idx = overlap.find(" ")
            if first_space_idx != -1:
                overlap = overlap[first_space_idx + 1:]

            # Find the last space to preserve complete words at the end
            last_space_idx = overlap.rfind(" ")
            if last_space_idx != -1:
                overlap = overlap[:last_space_idx]

            if overlap:
                overlapped.append(f"{overlap} {chunk}")
            else:
                overlapped.append(chunk)

        return overlapped

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        return " ".join(text.split())

def chunk_text(
        text: str,
        chunk_size_chars: int = 800,
        overlap_chars: int = 100,
        base_metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    """
    Backward-compatible helper.

    Internally uses FixedSizeCharacterChunker.
    """
    return FixedSizeCharacterChunker(
        chunk_size_chars=chunk_size_chars,
        overlap_chars=overlap_chars,
    ).chunk(
        text=text,
        base_metadata=base_metadata,
    )