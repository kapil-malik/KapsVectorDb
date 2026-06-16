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

#### M3.1 — `FileBackedVectorStore` ✅
- Persistent vector storage
- Append-only segment files
- Reloadable vector database

#### M3.2 — `MMapVectorStore` ✅
- Memory-mapped vector storage
- OS page cache behavior
- Larger-than-memory dataset handling

---

### Phase 4 — Approximate Nearest Neighbor (ANN)

Goal:
Understand how modern vector databases scale similarity search using partition-based and graph-based approximate indexing.

#### M4.1 — `IVFVectorStore` ✅
- Inverted File Index
- KMeans clustering
- `nlist` / `nprobe`
- Recall vs latency tradeoffs

#### M4.2 — `FlatNSWVectorStore` ✅
- Single-layer navigable small-world graph
- Best-first graph traversal
- `m` and `ef_search`
- Graph connectivity vs recall tradeoffs

#### M4.3 — `HNSWVectorStore` ✅
- Hierarchical graph layers
- Probabilistic level assignment
- Entry-point based search
- Layer-by-layer greedy descent
- Bottom-layer ef_search refinement

#### M4.4 — ANN Instrumentation & Diagnostics ✅
- Track visited nodes
- Track candidate expansions
- Track vectors scanned
- Track graph hops
- Track IVF clusters scanned
- Compare recall vs actual work done

#### M4.5 — ANN Visualization ⏳
- Small-dataset graph visualization
- NSW/HNSW traversal path visualization
- IVF cluster visualization
- Recall/latency/work curves

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
