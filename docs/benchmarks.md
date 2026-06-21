# Benchmarks

This project benchmarks vector-store internals, retriever latency, persistence
behavior, ANN recall/latency tradeoffs, and retrieval quality against labeled
relevance judgments.

## Benchmark Docs

| Benchmark | Script | Focus | Doc |
|---|---|---|---|
| Vector Store Benchmarks | `benchmarks/benchmark_store.py` | Insert/search latency for exact and persistent stores | [benchmarks/vector_store_benchmarks.md](benchmarks/vector_store_benchmarks.md) |
| Retriever Benchmark | `benchmarks/benchmark_retriever.py` | End-to-end semantic retrieval latency | [benchmarks/retriever_benchmark.md](benchmarks/retriever_benchmark.md) |
| Tombstones and Compaction | `benchmarks/benchmark_store.py` | Logical deletes vs physical storage size | [benchmarks/tombstones_compaction.md](benchmarks/tombstones_compaction.md) |
| PDF ANN Benchmark | `benchmarks/benchmark_pdf_ann_stores.py` | ANN recall vs latency against an exact vector-search baseline | [benchmarks/pdf_ann_benchmark.md](benchmarks/pdf_ann_benchmark.md) |
| Retrieval Quality Benchmark | `benchmarks/benchmark_retrieval_quality.py` | Retrieved results vs human-labeled qrels (BEIR datasets) | [benchmarks/retrieval_quality_benchmark.md](benchmarks/retrieval_quality_benchmark.md) |

## How to Read These Benchmarks

**Exact vector-search benchmarks** measure raw store performance: insert throughput,
search latency percentiles, and how internal layout choices affect each. These
benchmarks do not evaluate result quality — they measure speed only.

**ANN benchmarks** compare approximate stores against exact vector search used as a
recall baseline. A result is counted as recalled if it matches what exact search would
have returned for the same query. This is not the same as ground-truth semantic
relevance — exact search can still return wrong answers for a query, and ANN recall
against it measures algorithmic fidelity, not answer quality.

**Retrieval quality benchmarks** compare retrieved results against human-labeled
relevance judgments (qrels). These are a stronger measure of answer quality but
require a labeled dataset. They are tracked separately from the ANN recall benchmarks.

Exact search is often used as a baseline in ANN benchmarks because it is the
deterministic upper bound on recall for a given embedding model and distance metric,
not because its results are always semantically correct.