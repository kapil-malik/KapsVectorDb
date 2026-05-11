from pathlib import Path

from pypdf import PdfReader


def load_pdf_pages(file_path: str | Path) -> list[tuple[int, str]]:
    """
    Returns list of:
    (page_number, page_text)
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))

    pages: list[tuple[int, str]] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            pages.append((i + 1, text))

    return pages