# Vector Store Benchmarks

Benchmarks insert throughput and search latency across the project's exact and
persistent store implementations using `benchmark_store.py`.

## What It Benchmarks

- Vector insertion performance (throughput, latency)
- Vector similarity search latency (avg, p50, p95, p99)

## Run

```bash
poetry run python benchmarks/benchmark_store.py \
  --store buffered-matrix \
  --records 10000 \
  --dim 384 \
  --queries 100 \
  --top-k 5
```

## Arguments

| Argument | Description |
|---|---|
| `--store` | Store implementation: `naive`, `normalized`, `matrix`, `buffered-matrix` |
| `--records` | Number of vectors to insert |
| `--dim` | Vector dimensions |
| `--queries` | Number of search queries to run |
| `--top-k` | Number of top results to retrieve |

## Metrics Reported

- Insert throughput (records/sec)
- Average search latency
- p50 / p95 / p99 search latency
- Queries per second

## Store Comparison

| Store | Insert Performance | Search Performance |
|---|---|---|
| `NaiveInMemVectorStore` | Fast | Slow |
| `NormalizedInMemVectorStore` | Slightly slower | Much faster |
| `MatrixBackedInMemVectorStore` | Extremely slow inserts | Extremely fast search |
| `BufferedMatrixInMemVectorStore` | Fast inserts | Extremely fast search |

## Key Learnings

- **Vector normalization** significantly improves search efficiency by enabling dot
  product to substitute for cosine similarity without the per-query norm division.
- **Data layout matters**: a dense matrix representation allows NumPy vectorized
  operations that are orders of magnitude faster than Python-loop comparisons.
- **NumPy vectorization** outperforms Python loops by a large margin on vector search.
- **Repeated `np.vstack`** creates severe write-path bottlenecks by reallocating the
  entire matrix on every insert.
- **Buffered batching** amortizes matrix reallocation across many inserts, preserving
  fast search while dramatically improving insert throughput.