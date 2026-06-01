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
## PDF ANN Store Benchmark

A benchmark was run to compare exact search, IVF, and Flat NSW on a real PDF-based retrieval workload.

### Benchmark Setup

```commandline
poetry run python benchmarks/benchmark_pdf_ann_stores.py \
  --pdf The_DynamoDb_Book.pdf \
  --queries-file ./benchmarks/data/pdf_queries.txt \
  --output-csv ./benchmarks/results/pdf_ann_results.csv \
  --top-k 5 \
  --ivf-nlist 10,20,40 \
  --ivf-nprobe 1,5,10 \
  --nsw-m 8,16 \
  --nsw-ef-search 16,32,64
```

Dataset:
- PDF: *The DynamoDB Book*
- Records: 881 PDF chunks
- Queries: 50 semantic queries
- Embedding model: Sentence Transformer
- `top_k`: 5

Stores compared:
- `BufferedMatrixInMemVectorStore` as the exact baseline
- `IVFVectorStore`
- `FlatNSWVectorStore`

### Results

| Store | Parameters | Prepare Time | p95 Search Latency | Avg Recall@5 |
|---|---|---:|---:|---:|
| Exact | `{}` | 0.003 sec | 0.431 ms | 1.000 |
| IVF | `nlist=10, nprobe=1` | 0.136 sec | 0.060 ms | 0.676 |
| IVF | `nlist=10, nprobe=5` | 0.029 sec | 0.139 ms | 0.996 |
| IVF | `nlist=10, nprobe=10` | 0.033 sec | 0.127 ms | 1.000 |
| IVF | `nlist=20, nprobe=1` | 0.031 sec | 0.047 ms | 0.524 |
| IVF | `nlist=20, nprobe=5` | 0.032 sec | 0.091 ms | 0.940 |
| IVF | `nlist=20, nprobe=10` | 0.031 sec | 0.131 ms | 0.992 |
| IVF | `nlist=40, nprobe=1` | 0.068 sec | 0.037 ms | 0.528 |
| IVF | `nlist=40, nprobe=5` | 0.066 sec | 0.075 ms | 0.908 |
| IVF | `nlist=40, nprobe=10` | 0.067 sec | 0.090 ms | 0.972 |
| Flat NSW | `m=8, ef_search=16` | 0.306 sec | 0.096 ms | 0.624 |
| Flat NSW | `m=8, ef_search=32` | 0.293 sec | 0.156 ms | 0.784 |
| Flat NSW | `m=8, ef_search=64` | 0.294 sec | 0.262 ms | 0.892 |
| Flat NSW | `m=16, ef_search=16` | 0.460 sec | 0.140 ms | 0.888 |
| Flat NSW | `m=16, ef_search=32` | 0.437 sec | 0.219 ms | 0.952 |
| Flat NSW | `m=16, ef_search=64` | 0.441 sec | 0.335 ms | 0.996 |

### Key Learnings

IVF shows the expected `nprobe` tradeoff:

```text
lower nprobe -> lower latency, lower recall
higher nprobe -> higher recall, higher latency
```

Flat NSW shows the expected graph-search tradeoff:
```text
higher ef_search -> better recall, higher latency
higher m -> denser graph, better recall, higher insert cost
```

The best IVF configurations achieved near-exact recall with much lower p95 latency than exact search.

Flat NSW also reached near-exact recall, but with higher preparation time because the current implementation uses brute-force neighbor discovery during insert.

This benchmark demonstrates the difference between two ANN strategies:
```text
IVF      -> partition then scan
Flat NSW -> navigate then refine
```