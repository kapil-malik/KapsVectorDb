# KapsVectorDb

A fresh attempt at vector database design. This project implements an in-memory vector database with support for semantic search, PDF ingestion, and different chunking strategies.

## Project Structure

The main source code is in `src/vectordb/`:
- `stores/`: Nine vector store implementations — exact-search (Naive, Normalized, Matrix-backed, Buffered), persistent (FileBacked, MMap), and ANN (IVF, FlatNSW, HNSW).
- `embeddings/`: Embedding models (SentenceTransformer support).
- `ingestion/`: Text chunking and PDF loading utilities (chunkers, PDF loader).
- `retrieval/`: Semantic text retrieval using embeddings.
- `models.py`: Core data structures (`VectorRecord`, `SearchResult`, `SearchDiagnostics`).

Benchmarks live in `benchmarks/` and visualization scripts in `visualizations/`.

## Quick Start

### Installation

Run the project dependencies and tools:
```bash
poetry install
```

## Main Entry Points
This project provides several demonstration scripts located in `src/vectordb/`. Run them via Poetry using the module flag (`-m`) so Python resolves `vectordb` from `src/`.

**1) main.py — Basic Vector Store Test**

Tests and compares different in-memory vector store implementations with a simple similarity search.

Run:
```bash
poetry run python -m vectordb.main
```

What it demonstrates:
- NaiveInMemVectorStore
- NormalizedInMemVectorStore
- MatrixBackedInMemVectorStore
- BufferedMatrixInMemVectorStore
- FileBackedVectorStore
- MMapVectorStore
- IVFVectorStore
- FlatNSWVectorStore
- HNSWVectorStore

**2) main_chunker_compare.py — Chunking Strategy Comparison**

Compares two text chunking strategies (fixed-size and recursive) on a sample text and prints chunk metadata and text.

Run:
```bash
poetry run python -m vectordb.main_chunker_compare
```

What it demonstrates:
- `FixedSizeCharacterChunker`: Character-count based chunking.
- `RecursiveTextChunker`: Tries to preserve paragraphs/sentences/words.

**3) main_retriever.py — Semantic Text Retrieval Demo**

Demonstrates semantic search using a `SentenceTransformer` embedding model over in-memory text chunks.

Run:
```bash
poetry run python -m vectordb.main_retriever
```

What it demonstrates:
- Initializes `SentenceTransformerEmbeddingModel`.
- Indexes a handful of sample chunks with metadata.
- Runs a semantic query and prints top results with scores and metadata.
- Demonstrates metadata filtering on results.
- Covers: NaiveInMemVectorStore, NormalizedInMemVectorStore, MatrixBackedInMemVectorStore, BufferedMatrixInMemVectorStore, FlatNSWVectorStore, HNSWVectorStore.

**4) main_pdf_retriever.py — PDF-based Semantic Search**

Loads a PDF, chunk it, index with embeddings, and search over the content.

Run (example):
```bash
poetry run python -m vectordb.main_pdf_retriever --pdf ./The_DynamoDb_Book.pdf --query "partitioning" --top-k 3
poetry run python -m vectordb.main_pdf_retriever --pdf ./The_DynamoDb_Book.pdf --query "partitioning" --store hnsw
```

Arguments:
- `--pdf` (required): Path to the PDF file to index.
- `--query` (required): Search query string.
- `--top-k` (optional): Number of top results to return (default: 5).
- `--store` (optional): Vector store to use — `naive`, `normalized`, `matrix`, `buffered`, `ivf`, `flat_nsw`, `hnsw` (default: `buffered`).

What it does:
- Uses `chunks_from_pdf` to split the PDF into chunks.
- Uses `SentenceTransformerEmbeddingModel` to create embeddings.
- Indexes chunks into the chosen vector store.
- Performs semantic search and prints top-k matches.

## Roadmap Summary

- Phase 1 — Exact-search vector store internals ✅
- Phase 2 — Semantic retrieval / RAG layer ✅
- Phase 3 — Persistence and metadata filtering ✅
- Phase 4 — Approximate nearest neighbor indexes (IVF / HNSW) ✅
- Phase 5 — Retrieval quality evaluation ⏳

See [docs/roadmap.md](docs/roadmap.md) for the full milestone breakdown.

## Development

Running Tests:
```bash
poetry run pytest
```

Code formatting and static checks:
```bash
poetry run black src/
poetry run isort src/
poetry run mypy src/
```

Notes and Tips

- Use the module runner form (`python -m vectordb.<module>`) from the project root so `src/` is source-rooted for Poetry. Example:
    - `poetry run python -m vectordb.main`
- If you prefer running a script file directly, ensure `PYTHONPATH=src` is set, e.g.:
    - `PYTHONPATH=src poetry run python src/vectordb/main.py`
- `pyproject.toml` already includes `packages = [{include = "vectordb", from = "src"}]`, so installing via `poetry install` will correctly register `vectordb` for module runs.
- If you add or change dependencies, re-run:
    - `poetry install` or `poetry update` as appropriate.

### Dependencies 

- `python` >= 3.12
- `numpy`
- `sentence-transformers` (for embeddings)
- `pypdf` (for PDF ingestion)
- `scikit-learn` (for PCA in visualizations)
- Dev: `pytest`, `black`, `isort`, `mypy`, `tqdm`, `matplotlib`, `pandas`

## Documentation

Detailed project documentation:

- [Architecture Evolution](docs/architecture.md)
- [Benchmarking & Learnings](docs/benchmarks.md)
- [Project Roadmap](docs/roadmap.md)
- [ANN Visualizations](docs/visualizations.md)