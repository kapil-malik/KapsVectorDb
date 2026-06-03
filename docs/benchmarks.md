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

A benchmark was run to compare exact search, IVF, Flat NSW, and HNSW on a real PDF-based retrieval workload.

> **Note:**
> This is still a relatively small ANN benchmark corpus (~881 vectors).
> Exact search is already sub-millisecond at this scale, so ANN structures like HNSW do not yet fully demonstrate their large-scale advantages. The primary goal of this benchmark is learning ANN behavior and recall/latency tradeoffs, not production-scale performance claims.

### Benchmark Setup

```commandline
poetry run python benchmarks/benchmark_pdf_ann_stores.py \
  --pdf The_DynamoDb_Book.pdf \
  --queries-file ./benchmarks/data/pdf_queries.txt \
  --output-csv ./benchmarks/results/pdf_ann_results_2.csv \
  --top-k 5 \
  --ivf-nlist 10,20,40 \
  --ivf-nprobe 1,5,10 \
  --nsw-m 8,16 \
  --nsw-ef-search 16,32,64 \
  --hnsw-m 8,16 \
  --hnsw-ef-construction 32,64,128 \
  --hnsw-ef-search 16,32,64 \
  --hnsw-level-multiplier 1.0
```

Dataset:

* PDF: *The DynamoDB Book*
* Records: 881 PDF chunks
* Queries: 50 semantic queries
* Embedding model: Sentence Transformer
* `top_k`: 5

Stores compared:

* `BufferedMatrixInMemVectorStore` as the exact baseline
* `IVFVectorStore`
* `FlatNSWVectorStore`
* `HNSWVectorStore`

### Results

| Store    | Parameters                                | Prepare Time | p95 Search Latency | Avg Recall@5 |
| -------- | ----------------------------------------- | -----------: | -----------------: | -----------: |
| Exact    | `{}`                                      |    0.003 sec |           0.431 ms |        1.000 |
| IVF      | `nlist=10, nprobe=1`                      |    0.136 sec |           0.060 ms |        0.676 |
| IVF      | `nlist=10, nprobe=5`                      |    0.029 sec |           0.139 ms |        0.996 |
| IVF      | `nlist=10, nprobe=10`                     |    0.033 sec |           0.127 ms |        1.000 |
| IVF      | `nlist=20, nprobe=1`                      |    0.031 sec |           0.047 ms |        0.524 |
| IVF      | `nlist=20, nprobe=5`                      |    0.032 sec |           0.091 ms |        0.940 |
| IVF      | `nlist=20, nprobe=10`                     |    0.031 sec |           0.131 ms |        0.992 |
| IVF      | `nlist=40, nprobe=1`                      |    0.068 sec |           0.037 ms |        0.528 |
| IVF      | `nlist=40, nprobe=5`                      |    0.066 sec |           0.075 ms |        0.908 |
| IVF      | `nlist=40, nprobe=10`                     |    0.067 sec |           0.090 ms |        0.972 |
| Flat NSW | `m=8, ef_search=16`                       |    0.306 sec |           0.096 ms |        0.624 |
| Flat NSW | `m=8, ef_search=32`                       |    0.293 sec |           0.156 ms |        0.784 |
| Flat NSW | `m=8, ef_search=64`                       |    0.294 sec |           0.262 ms |        0.892 |
| Flat NSW | `m=16, ef_search=16`                      |    0.460 sec |           0.140 ms |        0.888 |
| Flat NSW | `m=16, ef_search=32`                      |    0.437 sec |           0.219 ms |        0.952 |
| Flat NSW | `m=16, ef_search=64`                      |    0.441 sec |           0.335 ms |        0.996 |
| HNSW     | `m=8, ef_construction=32, ef_search=16`   |    0.418 sec |           0.171 ms |        0.700 |
| HNSW     | `m=8, ef_construction=32, ef_search=32`   |    0.418 sec |           0.245 ms |        0.836 |
| HNSW     | `m=8, ef_construction=32, ef_search=64`   |    0.420 sec |           0.388 ms |        0.944 |
| HNSW     | `m=8, ef_construction=64, ef_search=16`   |    0.488 sec |           0.173 ms |        0.724 |
| HNSW     | `m=8, ef_construction=64, ef_search=32`   |    0.487 sec |           0.253 ms |        0.852 |
| HNSW     | `m=8, ef_construction=64, ef_search=64`   |    0.491 sec |           0.397 ms |        0.948 |
| HNSW     | `m=8, ef_construction=128, ef_search=16`  |    0.633 sec |           0.177 ms |        0.728 |
| HNSW     | `m=8, ef_construction=128, ef_search=32`  |    0.631 sec |           0.256 ms |        0.860 |
| HNSW     | `m=8, ef_construction=128, ef_search=64`  |    0.636 sec |           0.401 ms |        0.952 |
| HNSW     | `m=16, ef_construction=32, ef_search=16`  |    0.611 sec |           0.228 ms |        0.904 |
| HNSW     | `m=16, ef_construction=32, ef_search=32`  |    0.609 sec |           0.316 ms |        0.960 |
| HNSW     | `m=16, ef_construction=32, ef_search=64`  |    0.614 sec |           0.474 ms |        0.988 |
| HNSW     | `m=16, ef_construction=64, ef_search=16`  |    0.772 sec |           0.233 ms |        0.916 |
| HNSW     | `m=16, ef_construction=64, ef_search=32`  |    0.768 sec |           0.323 ms |        0.976 |
| HNSW     | `m=16, ef_construction=64, ef_search=64`  |    0.775 sec |           0.481 ms |        0.992 |
| HNSW     | `m=16, ef_construction=128, ef_search=16` |    1.054 sec |           0.236 ms |        0.920 |
| HNSW     | `m=16, ef_construction=128, ef_search=32` |    1.047 sec |           0.327 ms |        0.984 |
| HNSW     | `m=16, ef_construction=128, ef_search=64` |    1.058 sec |           0.486 ms |        0.992 |

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

HNSW shows the expected hierarchical graph behavior:

```text
higher ef_search        -> higher recall, higher latency
higher ef_construction  -> better graph quality, slower inserts
higher m                -> denser graph connectivity, higher recall
```

The best IVF configurations achieved near-exact recall with much lower p95 latency than exact search.

Flat NSW achieved near-exact recall using a single-layer graph traversal strategy.

HNSW achieved similarly high recall using hierarchical graph navigation, but with higher preparation cost because graph construction is more complex and multi-layered.

At this relatively small dataset size, IVF currently provides the best recall/latency tradeoff. HNSW is expected to become more advantageous at much larger corpus sizes where hierarchical navigation reduces search cost significantly.

This benchmark demonstrates three distinct ANN philosophies:

```text
IVF       -> partition then scan
Flat NSW  -> navigate then refine
HNSW      -> hierarchical navigation then refine
```
