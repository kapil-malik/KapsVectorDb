# Vector Store Benchmarks

Benchmarks insert throughput and search latency across all exact and persistent store
implementations using `benchmark_store.py`.

## What It Benchmarks

- Vector insertion throughput (records/sec) and total insert time
- Vector similarity search latency (avg, p50, p95, p99) and queries/sec

## Run

Single store:

```bash
poetry run python -m benchmarks.benchmark_store \
  --store buffered-matrix \
  --records 10000 \
  --dim 384 \
  --queries 100 \
  --top-k 5
```

All exact stores (runs all 6 sequentially on identical data, writes a summary CSV):

```bash
poetry run python -m benchmarks.benchmark_store \
  --store all \
  --records 10000 \
  --dim 384 \
  --queries 100 \
  --top-k 5 \
  --clean-file-store
```

## Arguments

| Argument | Description |
|---|---|
| `--store` | `naive`, `normalized`, `matrix`, `buffered-matrix`, `file`, `mmap`, or `all` |
| `--records` | Number of vectors to insert |
| `--dim` | Vector dimensions |
| `--queries` | Number of search queries to run |
| `--top-k` | Number of top results to retrieve |
| `--clean-file-store` | Wipe and recreate data directories for `file` and `mmap` stores |
| `--output-csv` | Override summary CSV path (default: `benchmarks/results/store_comparison.csv`) |

Soft-delete and compaction benchmarking is handled separately by `benchmark_tombstones.py`.

When `--store all` is used, all stores are benchmarked on the **same pre-generated
record and query vectors**, ensuring a fair comparison.

## Results

10,000 records · 384 dimensions · 100 queries · top-k 5

| Store | Insert time (s) | Insert throughput (rec/s) | Avg latency (ms) | p50 (ms) | p95 (ms) | p99 (ms) | QPS |
|---|---|---|---|---|---|---|---|
| `naive` | 0.003 | 3,795,967 | 31.92 | 28.83 | 33.20 | 94.71 | 31 |
| `normalized` | 0.029 | 349,889 | 14.21 | 10.94 | 14.86 | 74.66 | 70 |
| `matrix` | 6.508 | 1,537 | 0.573 | 0.462 | 0.825 | 1.121 | 1,744 |
| `buffered-matrix` | 0.040 | 247,595 | 0.860 | 0.875 | 1.033 | 1.136 | 1,163 |
| `file` | 0.070 | 141,955 | 0.550 | 0.420 | 0.823 | 1.198 | 1,818 |
| `mmap` | 0.071 | 139,934 | 0.397 | 0.379 | 0.489 | 0.979 | 2,517 |

## Key Learnings

### Normalization shifts work to the write path

Normalizing vectors at insert time converts cosine similarity search into a plain dot
product. The extra work at insert (one norm + divide per vector) pays off on every
subsequent query — `normalized` is 2.25× faster than `naive` at search with no change
to the search algorithm.

### Data layout dominates search latency

Storing vectors in a contiguous NumPy matrix and issuing a single `matrix @ query`
call instead of a Python loop over individual vectors drops avg search latency from
~14ms (dict-based) to sub-millisecond. The CPU can exploit BLAS/SIMD on contiguous
float32 memory in a way a Python loop never can.

### `argpartition` over `argsort` for top-k retrieval

Finding the top-k results from N scores does not require a full sort. `np.argsort` is
O(N log N); `np.argpartition` is O(N). For top-k=5 out of 10k vectors, `argsort`
examines ~133k comparisons; `argpartition` with a 4× candidate multiplier (top-20
candidates) needs only ~10k. All matrix-backed stores use `argpartition`. This was an
initial inconsistency between `matrix` (which used `argsort`) and `buffered-matrix`
(which always used `argpartition`) — both now use `argpartition` for consistent
semantics and performance.

### Buffered inserts amortise matrix reallocation

`MatrixBackedInMemVectorStore` calls `np.vstack` on every insert, reallocating the
entire matrix each time — O(N²) total work, taking 6.5 s for 10k records.
`BufferedMatrixInMemVectorStore` accumulates inserts in a list and flushes to the
matrix in batches of 1024, reducing insert time to 0.04 s (160× faster) while keeping
search latency in the same range.

### Unflushed buffer adds per-query overhead

`buffered-matrix` flushes every 1024 inserts. After 10k inserts, ~784 records remain
in the pending buffer. Every search call must `np.vstack` that buffer, run a second
matrix multiply, and merge results — a repeated allocation on the hot query path.
After `file` calls `save()` (which flushes the buffer to disk), its search path only
touches the single clean matrix, which is why `file` (0.55ms) is faster than
`buffered-matrix` (0.86ms) despite adding file I/O on the write path.

### MMapVectorStore is the fastest searcher once warm

`mmap` loads vectors via `mmap_mode="r"`, letting the OS page cache back the matrix
rather than copying it fully into Python's heap. After the five warmup queries the
benchmark runs before measurement, the relevant pages are hot in the OS cache. This
produces the lowest p50 and p99 of all stores (0.38ms / 0.98ms) despite an insert path
that goes through a `FileBackedVectorStore` intermediary.

### MMapVectorStore requires a two-phase write pattern

`MMapVectorStore` is read-only — `insert()` is not supported. The benchmark populates
it by inserting into a `FileBackedVectorStore` pointed at the same data directory, then
calling `save()` to flush, and finally constructing a fresh `MMapVectorStore` on the
same files. `MMapVectorStore.__init__` calls `load()` automatically, so the data is
immediately available for search without a separate `load()` call.
