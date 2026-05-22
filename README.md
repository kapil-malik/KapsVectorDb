# KapsVectorDb

A fresh attempt at vector database design. This project implements an in-memory vector database with support for semantic search, PDF ingestion, and different chunking strategies.

## Project Structure

The main source code is in `src/vectordb/`:
- `stores/`: Multiple in-memory vector store implementations (Naive, Normalized, Matrix-backed, Buffered).
- `embeddings/`: Embedding models (SentenceTransformer support).
- `ingestion/`: Text chunking and PDF loading utilities (chunkers, PDF loader).
- `retrieval/`: Semantic text retrieval using embeddings.
- `models.py`: Core data structures (`VectorRecord`, `SearchResult`).

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

What it does:
- Initializes `SentenceTransformerEmbeddingModel`.
- Indexes a handful of sample chunks with metadata.
- Runs a semantic query and prints top results with scores and metadata.

**4) main_pdf_retriever.py — PDF-based Semantic Search**

Loads a PDF, chunk it, index with embeddings, and search over the content.

Run (example):
```bash
poetry run python -m vectordb.main_pdf_retriever --pdf ./The_DynamoDb_Book.pdf --query "partitioning" --top-k 3
```

Arguments:
- `--pdf` (required): Path to the PDF file to index.
- `--query` (required): Search query string.
- `--top-k` (optional): Number of top results to return (default: 5).

What it does:
- Uses `chunks_from_pdf` to split the PDF into chunks.
- Uses `SentenceTransformerEmbeddingModel` to create embeddings.
- Indexes chunks into `BufferedMatrixInMemVectorStore`.
- Performs semantic search and prints top-k matches.

## Architecture Evolution

This project intentionally evolves through multiple vector store implementations to explore the tradeoffs behind real-world vector databases.

### 1. `NaiveInMemVectorStore`

Baseline implementation.

- Stores vectors in a Python dictionary.
- Uses brute-force cosine similarity search.
- Computes similarity against every vector at query time.

Search complexity:

```text
O(N × D)
```

Where:
- `N` = number of vectors
- `D` = vector dimensions

This implementation prioritizes correctness and simplicity over performance.

---

### 2. `NormalizedInMemVectorStore`

Optimization over the naive implementation.

Key idea:
- Normalize vectors during insert.
- Normalize query vector during search.
- Use dot product similarity instead of full cosine similarity.

Instead of:

```text
cosine_similarity(a, b) = dot(a, b) / (||a|| × ||b||)
```

Search becomes:

```text
dot(normalized_query, normalized_vector)
```

This shifts work from the read path to the write path and significantly reduces search latency.

---

### 3. `MatrixBackedInMemVectorStore`

Introduces matrix-backed vector storage using NumPy.

Key idea:
- Store vectors in one contiguous matrix.
- Perform vectorized similarity computation.

Instead of:

```python
for vector in vectors:
    np.dot(query, vector)
```

Search becomes:

```python
scores = matrix @ query
```

This dramatically improves search performance using optimized native NumPy/BLAS operations.

Tradeoff:
- Insert performance becomes very poor because the matrix is repeatedly reallocated using `np.vstack(...)`.

---

### 4. `BufferedMatrixInMemVectorStore`

Improves the matrix-backed implementation using buffered inserts.

Key idea:
- Maintain:
    - a large immutable-ish matrix
    - a small mutable insert buffer
- Flush buffer into matrix periodically in batches.

Benefits:
- Retains fast vectorized matrix search.
- Avoids repeated matrix reallocations on every insert.
- Greatly improves insert throughput.

This design loosely resembles:
- memtables + SSTables
- buffered write paths in LSM-based systems

---

## Benchmarking

The project includes benchmark scripts to compare vector store implementations and semantic retrieval performance.

### 1. `benchmark_store.py`

Benchmarks:
- vector insertion performance
- vector similarity search latency

Run:

```bash
poetry run python benchmarks/benchmark_store.py \
  --store buffered-matrix \
  --records 10000 \
  --dim 384 \
  --queries 100 \
  --top-k 5
```

Arguments:
- `--store`: Store implementation (`naive`, `normalized`, `matrix`, `buffered-matrix`)
- `--records`: Number of vectors to insert
- `--dim`: Vector dimensions
- `--queries`: Number of search queries
- `--top-k`: Number of top results to retrieve

Metrics reported:
- insert throughput
- average latency
- p50 / p95 / p99 latency
- queries per second

---

### 2. `benchmark_retriever.py`

Benchmarks semantic text retrieval.

Measures:
- embedding generation
- chunk ingestion
- semantic search latency

Run:

```bash
poetry run python benchmarks/benchmark_retriever.py \
  --chunks 1024 \
  --queries 100 \
  --embedding-model sentence-transformer
```

Arguments:
- `--chunks`: Number of text chunks
- `--queries`: Number of semantic queries
- `--top-k`: Top results to retrieve
- `--embedding-model`: `fake` or `sentence-transformer`

This benchmark measures the full retrieval pipeline:

```text
text
→ embedding generation
→ vector search
→ chunk retrieval
```

---

## Benchmark Learnings So Far

Some early observations from experimentation:

| Store | Insert Performance | Search Performance |
|---|---|---|
| `NaiveInMemVectorStore` | Fast | Slow |
| `NormalizedInMemVectorStore` | Slightly slower | Much faster |
| `MatrixBackedInMemVectorStore` | Extremely slow inserts | Extremely fast search |
| `BufferedMatrixInMemVectorStore` | Fast inserts | Extremely fast search |

Key learnings:
- Vector normalization significantly improves search efficiency.
- Data layout matters enormously for vector search.
- Vectorized matrix operations outperform Python loops by a large margin.
- Repeated matrix reallocations (`np.vstack`) create severe write-path bottlenecks.
- Buffered batching dramatically improves insert throughput while preserving fast search.

## Roadmap

### Phase 1 — Vector Store Internals

Goal:
Understand how vector databases work internally by implementing progressively optimized exact-search vector stores from scratch.

#### M1.1 — `NaiveInMemVectorStore` ✅
- Basic in-memory vector store
- Brute-force cosine similarity search
- Exact top-k retrieval
- Baseline correctness implementation

#### M1.2 — `benchmark_store.py` ✅
- Insert/search benchmarking
- p50 / p95 / p99 latency measurement
- Throughput metrics
- Store comparison framework

#### M1.3 — `NormalizedInMemVectorStore` ✅
- Vector normalization during insert
- Dot-product similarity optimization
- Faster cosine similarity search

#### M1.4 — `MatrixBackedInMemVectorStore` ✅
- Dense matrix-backed vector storage
- Vectorized similarity computation using NumPy
- Extremely fast search using matrix multiplication

#### M1.5 — `BufferedMatrixInMemVectorStore` ✅
- Buffered inserts
- Batched matrix flushes
- Fast write path + fast search path
- LSM/memtable-inspired design ideas

#### M1.6 — Compare Benchmark Results ✅
- Compare latency/throughput tradeoffs
- Analyze insert vs search optimization tradeoffs
- Understand effects of:
    - normalization
    - vectorized compute
    - memory layout
    - batching

---

### Phase 2 — Semantic Retrieval / RAG Layer

Goal:
Build a semantic retrieval system on top of the vector store abstraction.

#### M2.1 — `EmbeddingModel` protocol ✅
- Pluggable embedding model abstraction
- Support for:
    - fake embeddings
    - SentenceTransformer embeddings
    - future OpenAI/Cohere/BGE models

#### M2.2 — `SemanticTextRetriever` ✅
- User-facing semantic retrieval abstraction
- Text → embeddings → vector search pipeline

#### M2.3 — Simple RAG ingestion from `list[str]` ✅
- Add/search plain text chunks
- Internal embedding generation

#### M2.4 — PDF chunk ingestion ✅
- PDF loading
- Text extraction
- Chunk generation with metadata

#### M2.5 — `benchmark_retriever.py` ✅
- Benchmark end-to-end retrieval pipeline
- Measure:
    - embedding latency
    - ingestion throughput
    - semantic retrieval latency

#### M2.6 — Better Chunkers ✅

##### M2.6.1 — `Chunker` protocol ✅
- Pluggable chunking abstraction

##### M2.6.2 — `FixedSizeCharacterChunker` ✅
- Character-count based chunking baseline

##### M2.6.3 — `RecursiveTextChunker` ✅
- Paragraph/sentence-aware chunking
- Better semantic coherence for RAG

##### M2.6.4 — Chunking comparison tooling ✅
- Compare chunking outputs visually
- Analyze chunk quality tradeoffs

#### M2.7 — Future Chunking Improvements ⏳
- Token-aware chunking
- Semantic chunking
- Document-structure-aware chunking

---

### Phase 3 — Persistence & Metadata

Goal:
Understand how real vector databases persist data and support filtered retrieval.

#### M3.0 — Metadata Filtering ✅
- Metadata-aware vector search
- Payload filtering
- Query-time filters

#### M3.1 — `FileBackedVectorStore` ⏳
- Persistent vector storage
- Append-only segment files
- Reloadable vector database

#### M3.2 — `MMapVectorStore` ⏳
- Memory-mapped vector storage
- OS page cache behavior
- Larger-than-memory dataset handling

---

### Phase 4 — Approximate Nearest Neighbor (ANN)

Goal:
Understand how modern vector databases scale similarity search.

#### M4.1 — `IVFVectorStore` ⏳
- Inverted File Index (IVF)
- Clustering-based ANN search
- Recall vs latency tradeoffs

#### M4.2 — `HNSWVectorStore` ⏳
- Hierarchical Navigable Small World graphs
- Graph-based ANN search
- Fast approximate nearest neighbor traversal

---

### Phase 5 — Retrieval Quality Evaluation

Goal:
Measure retrieval quality, not just latency/performance.

#### M5.1 — Quality Benchmarks ⏳
- Recall@k
- MRR (Mean Reciprocal Rank)
- Retrieval relevance evaluation
- Labeled query/chunk datasets

#### M5.2 — Hybrid Retrieval ⏳
- BM25 + vector search
- Semantic + keyword retrieval
- Hybrid ranking experiments

#### M5.3 — Reranking ⏳
- Cross-encoder reranking
- Multi-stage retrieval pipelines

---

### Long-Term Goals

- Distributed vector storage
- Segment compaction
- WAL and crash recovery
- Multi-tenant retrieval
- Vector compression / quantization
- Product Quantization (PQ)
- DiskANN-style search
- Hybrid structured + semantic retrieval
- Agentic retrieval pipelines
```

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
- Dev: `pytest`, `black`, `isort`, `mypy`, `tqdm`