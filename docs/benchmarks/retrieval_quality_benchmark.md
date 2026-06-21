# Retrieval Quality Benchmark

## Goal

Fast search is not enough for RAG. Retrieved documents must actually be relevant to
the query — a store that returns results in 1 ms is not useful if those results are
wrong.

This benchmark measures retrieval quality using labeled relevance judgments (qrels)
from the BEIR benchmark suite. Each query has a set of documents marked relevant by
human annotators or dataset curators. A retrieved result is evaluated against those
labels, not against what exact vector search would have returned.

This is the key distinction from the ANN benchmarks elsewhere in this project:

| Benchmark type | Measures against |
|---|---|
| ANN recall benchmark | Exact vector search results (algorithmic fidelity) |
| Retrieval quality benchmark | Human-labeled qrels (answer relevance) |

Exact vector search is used here as a quality baseline — it gives the best possible
retrieval quality a given embedding model can achieve on this dataset — not as the
relevance oracle. It can still return documents that are not labeled relevant.

## Dataset Format

Datasets are downloaded from BEIR and converted to a flat format using
`benchmarks/download_beir_dataset.py`.

**`corpus.jsonl`** — one document per line:
```json
{"id": "doc_id", "text": "document text (title + body concatenated)"}
```

**`queries.jsonl`** — one query per line:
```json
{"id": "query_id", "text": "query text"}
```

**`qrels.tsv`** — relevance judgments, tab-separated:
```
query_id\tdoc_id\trelevance
```
Relevance is a non-negative integer; scores > 0 are treated as relevant.

The download script guarantees that all qrel-referenced documents are present in the
output corpus even when `--max-corpus-docs` is used to cap the corpus size. This
preserves valid retrieval quality measurement on sampled subsets.

## Metrics

All metrics are computed at `top_k` (default 10) and averaged across queries.

| Metric | What it measures |
|---|---|
| `Precision@K` | Fraction of retrieved results that are relevant |
| `Recall@K` | Fraction of all relevant documents that were retrieved |
| `HitRate@K` | 1 if at least one relevant document was retrieved, else 0 |
| `MRR@K` | Mean Reciprocal Rank — rewards finding a relevant result early |
| `nDCG@K` | Normalized Discounted Cumulative Gain — rewards relevant results appearing high in the ranking |

**nDCG@K is the headline metric.** It accounts for both whether relevant documents
were retrieved and how high they ranked. A result returned at rank 1 contributes more
than the same result at rank 10.

## Benchmark Commands

### Download and convert a dataset

```bash
python -m benchmarks.download_beir_dataset \
  --dataset hotpotqa \
  --output-dir data/retrieval_quality/beir/hotpotqa_100k \
  --max-corpus-docs 100000 \
  --max-queries 10000
```

Key arguments for `download_beir_dataset.py`:

| Argument | Description |
|---|---|
| `--dataset` | BEIR dataset name (e.g. `scifact`, `fiqa`, `hotpotqa`) |
| `--output-dir` | Where to write `corpus.jsonl`, `queries.jsonl`, `qrels.tsv` |
| `--max-corpus-docs` | Cap corpus size; qrel-positive docs are always included |
| `--max-queries` | Sample N queries deterministically |
| `--split` | qrels split: `test` (default), `dev`, or `train` |
| `--sample-seed` | Random seed for deterministic sampling (default: 42) |

### Run the benchmark

Single-store sanity check:

```bash
python -m benchmarks.benchmark_retrieval_quality \
  --dataset data/retrieval_quality/beir/hotpotqa_100k \
  --store buffered \
  --embedding-model sentence-transformer \
  --embedding-cache-dir data/retrieval_quality/cache/hotpotqa_100k \
  --top-k 10
```

Full sweep across all stores:

```bash
python -m benchmarks.benchmark_retrieval_quality \
  --dataset data/retrieval_quality/beir/hotpotqa_100k \
  --store all \
  --embedding-model sentence-transformer \
  --embedding-cache-dir data/retrieval_quality/cache/hotpotqa_100k \
  --top-k 10 \
  --ivf-nlist "10,20,40" \
  --ivf-nprobe "2,5,10" \
  --nsw-m "8,16" \
  --nsw-ef-construction "32,64,128" \
  --nsw-ef-search "16,32,64" \
  --hnsw-m "8,16" \
  --hnsw-ef-construction "32,64,128" \
  --hnsw-ef-search "16,32,64" \
  --hnsw-level-multiplier 1.0 \
  --output-csv benchmarks/results/hotpotqa_100k_sweep.csv
```

Key arguments for `benchmark_retrieval_quality.py`:

| Argument | Description |
|---|---|
| `--dataset` | Path to converted dataset directory |
| `--store` | Store to benchmark: `buffered`, `ivf`, `flat_nsw`, `flat_nsw_v2`, `hnsw`, `all` |
| `--embedding-model` | `fake` (random) or `sentence-transformer` |
| `--embedding-cache-dir` | Cache directory for embeddings (avoids re-embedding) |
| `--top-k` | Number of results to retrieve per query |
| `--max-eval-queries` | Cap the number of evaluated queries |
| `--output-csv` | Write summary metrics to a CSV file |
| `--per-query-output-csv` | Write per-query results (single-config runs only) |

## HotpotQA 100k Results

**Dataset:** 100,000 corpus documents, 7,405 queries, `top_k=10`
**Embedding:** `sentence-transformers/all-MiniLM-L6-v2`

The table below shows selected representative configurations. See
`benchmarks/results/hotpotqa_100k_sweep.csv` for the full sweep.

| Store | Parameters | Avg latency | p95 latency | Recall@10 | MRR@10 | nDCG@10 | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| Buffered (exact) | `buffer_size=1024` | 4.97 ms | 5.51 ms | 0.707 | 0.859 | 0.694 | Full quality ceiling |
| IVF | `nlist=20, nprobe=5` | 9.69 ms | 13.20 ms | 0.657 | 0.819 | 0.650 | ~94% of nDCG, 2× slower |
| FlatNSW v2 | `m=16, ef_c=128, ef_search=64` | 1.19 ms | 1.70 ms | 0.564 | 0.714 | 0.562 | ~81% of nDCG, 4× faster |
| HNSW | `m=16, ef_c=128, ef_search=64` | 1.20 ms | 1.61 ms | 0.558 | 0.705 | 0.555 | ~80% of nDCG, 4× faster |

### Observations

**IVF** at this corpus size tends to be either comparable in speed to exact search
(higher nprobe settings) or faster at a significant quality cost (low nprobe). The
`nlist=20, nprobe=5` configuration recovers about 94% of nDCG at roughly 2× the
latency — which is not a latency win, but confirms quality is largely preserved.
Configurations like `nlist=40, nprobe=2` are 2–3× faster than exact search but drop
nDCG to ~0.53, a 24% quality loss.

**FlatNSW v2 and HNSW** both achieve approximately 4× lower query latency than exact
search but at a meaningful quality cost. Both drop to ~nDCG 0.55 at their best
configurations — about an 80% quality ceiling compared to the exact baseline. Raising
`m`, `ef_construction`, or `ef_search` all push recall higher, at higher latency or
build cost.

**Exact search** remains the quality leader at this scale. Its latency (~5 ms at
100k docs) is not prohibitive for many use cases, and the quality gap vs ANN stores is
real and measurable in labeled evaluations.

## Key Learnings

**Exact search is a strong quality baseline.**
At 100k documents, the exact buffered store achieves nDCG 0.694 at ~5 ms per query.
Any ANN configuration that substantially improves latency will lose some retrieval
quality — this is the fundamental ANN tradeoff, measurable here against real labels.

**IVF preserves quality well but provides modest latency wins at this scale.**
With reasonable nprobe settings, IVF recovers ~90–94% of nDCG vs exact. Aggressive
nprobe reduction (e.g. nprobe=2) gives 2–3× speedup but drops quality noticeably.

**Graph stores (FlatNSW v2, HNSW) provide the strongest latency gains but lose more quality.**
Both achieve ~4× faster query latency at their best-quality configurations, settling
around nDCG ~0.55–0.56. They are the right tradeoff when throughput is critical and
moderate quality loss is acceptable.

**Higher `m`, `ef_construction`, and `ef_search` all improve graph quality.**
- `m` controls graph connectivity — higher m improves recall but increases build time
  and graph memory
- `ef_construction` controls graph quality at build time — higher values improve
  neighborhood selection at the cost of slower inserts
- `ef_search` controls beam search width at query time — the most direct latency/recall
  lever at search time

**Retrieval quality is not the same as ANN recall.**
An ANN store can achieve 99% recall against exact vector search while still producing
poor nDCG against qrels — because the exact search results themselves are not perfectly
aligned with human relevance judgments. Labeled evaluation is the more meaningful
signal for RAG use cases.

**For RAG, labeled retrieval metrics matter more than ANN recall alone.**
ANN recall measures whether approximate search reproduces exact search. Labeled metrics
measure whether retrieved documents are actually useful. Both matter, but nDCG against
qrels is closer to the user-facing quality signal.

## Smaller BEIR Runs

Before running the 100k HotpotQA sweep, two smaller datasets were used for development
and sanity checking.

**SciFact** (5,183 documents, 300 queries) — useful as a fast sanity check. At this
scale, exact search is already sub-millisecond and the ANN stores show no meaningful
latency advantage. Retrieval quality tradeoffs are visible but not as pronounced as at
larger scale.

**FiQA** (57,638 documents, 648 queries) — a useful intermediate dataset. Large enough
to show ANN latency advantages over exact search (~5× faster for graph stores), and
representative enough to observe quality tradeoffs. Used as the primary development
dataset before scaling to HotpotQA 100k.

**HotpotQA 100k** is the main documented result. At this scale, the latency difference
between exact and ANN stores is meaningful (~5 ms vs ~1.2 ms), and the quality
tradeoffs are large enough to reason about clearly.

## What This Does Not Measure Yet

- **Hybrid retrieval** — combining BM25 keyword search with vector search
- **Reranking** — a second-stage model to reorder the retrieved candidates
- **Answer-generation quality** — whether a language model produces correct answers
  given the retrieved context
- **Multi-stage retrieval pipelines** — retrieve-then-filter or retrieve-then-rerank
  chains