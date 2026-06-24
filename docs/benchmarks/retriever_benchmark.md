# Retriever Benchmark

Benchmarks the full end-to-end semantic retrieval pipeline using
`benchmark_retriever.py`.

## What It Benchmarks

The full text retrieval pipeline:

```text
text
→ embedding generation
→ vector search
→ chunk retrieval
```

Measures:
- Embedding generation latency
- Chunk ingestion throughput
- Semantic search latency

## Run

```bash
poetry run python -m benchmarks.benchmark_retriever \
  --chunks 1024 \
  --queries 100 \
  --embedding-model sentence-transformer
```

## Arguments

| Argument | Description |
|---|---|
| `--chunks` | Number of text chunks to ingest |
| `--queries` | Number of semantic queries to run |
| `--top-k` | Number of top results to retrieve |
| `--embedding-model` | `fake` (random vectors) or `sentence-transformer` |

## Metrics Reported

- Embedding throughput (chunks/sec, queries/sec)
- Insert throughput
- Average and percentile search latency

## What This Is Useful For

This benchmark reflects real-world retrieval cost more closely than the raw store
benchmark because it includes embedding generation. It is useful for comparing the
relative cost of embedding vs search as the dataset scales, and for checking that the
search component is not a bottleneck relative to the embedding step.

Use `--embedding-model fake` for pure store profiling without model inference cost.
Use `--embedding-model sentence-transformer` for realistic end-to-end timing.