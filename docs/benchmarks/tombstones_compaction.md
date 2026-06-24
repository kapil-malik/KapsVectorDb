# Tombstones and Compaction Benchmark

A benchmark exploring how tombstones and compaction affect search performance in the
file-backed store.

## Benchmark Setup

```bash
poetry run python -m benchmarks.benchmark_tombstones \
  --records 100000 \
  --delete-count 60000 \
  --dim 384 \
  --queries 1000 \
  --top-k 5 \
  --clean
```

## Workflow

1. Insert 40k vectors → benchmark search latency
2. Insert 60k more vectors (total = 100k) → benchmark search latency
3. Delete 60k vectors (logical count drops to 40k) → benchmark before compaction
4. Run compaction → benchmark after compaction

## Results

| Stage | Logical Records | Physical Matrix Rows | p95 Search Latency |
|---|---:|---:|---:|
| After first 40k insert | 40k | 40k | ~2.1 ms |
| After total 100k insert | 100k | 100k | ~5.1 ms |
| After deleting 60k | 40k | 100k | ~4.9 ms |
| After compaction | 40k | 40k | ~2.0 ms |

## Key Learnings

```text
Delete reduces logical records.
Compaction reduces physical records.
```

Before compaction, deleted records still physically exist in `vectors.npy` and
`records.jsonl`. Search still scans the full vector matrix — tombstones are only
used to skip deleted records during candidate selection, not to reduce scan cost.

Compaction physically rewrites both files using only live records. This reduces:
- the size of the vector matrix
- the number of rows scanned per query
- search latency

The results confirm that exact vector search scales approximately linearly with the
number of **physically stored** vectors, not the logical record count:

```text
O(N × D)
```

Where `N` is the number of physical matrix rows and `D` is the vector dimensionality.
After compaction returns the physical row count to 40k, latency returns to ~2.0 ms —
matching the original 40k insert benchmark.