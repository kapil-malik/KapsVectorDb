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

## Tombstones and Compaction Benchmark

A benchmark was run to understand the impact of tombstones and compaction on search performance.

### Benchmark Setup

```bash
poetry run python benchmarks/benchmark_store.py \
  --store file \
  --records 100000 \
  --dim 384 \
  --queries 1000 \
  --top-k 5 \
  --delete-count 60000 \
  --compact \
  --clean-file-store
```

### Benchmark Workflow

1. Insert 40k vectors
2. Benchmark search latency
3. Insert 60k additional vectors (total = 100k)
4. Benchmark search latency
5. Delete 60k vectors (logical count = 40k)
6. Benchmark search latency before compaction
7. Run compaction
8. Benchmark search latency after compaction

### Results

| Stage | Logical Records | Physical Matrix Rows | p95 Search Latency |
|---|---:|---:|---:|
| After first 40k insert | 40k | 40k | ~2.1 ms |
| After total 100k insert | 100k | 100k | ~5.1 ms |
| After deleting 60k | 40k | 100k | ~4.9 ms |
| After compaction | 40k | 40k | ~2.0 ms |

### Key Learnings

Logical deletes do not immediately improve search performance.

Before compaction:
- deleted records still physically exist inside:
    - `vectors.npy`
    - `records.jsonl`
- search still scans the full vector matrix
- tombstones are only used to skip deleted records during candidate selection

Compaction physically rewrites:
- `records.jsonl`
- `vectors.npy`

using only live records.

This reduces:
- vector matrix size
- scan cost
- search latency

This experiment demonstrates an important storage-engine principle:

```text
Delete reduces logical records.
Compaction reduces physical records.
```

The benchmark also confirms that exact vector search scales approximately linearly with the number of stored vectors:

```text
O(N × D)
```

Where:
- `N` = number of vectors
- `D` = vector dimensions