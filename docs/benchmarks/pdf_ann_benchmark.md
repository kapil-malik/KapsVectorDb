# PDF ANN Store Benchmark

Compares exact search, IVF, Flat NSW, and HNSW on a real PDF-based retrieval
workload. Recall is measured against exact vector search as the baseline.

> **Corpus size note:**
> This is a relatively small ANN benchmark corpus (~881 vectors). Exact search is
> already sub-millisecond at this scale, so ANN structures like HNSW do not yet fully
> demonstrate their large-scale advantages. The goal here is learning ANN behavior and
> recall/latency tradeoffs, not production-scale performance claims.
>
> **HNSW randomness note:** HNSW results can vary between runs due to random level
> assignment during graph construction. The random max-level of inserted nodes affects
> both the graph topology and the number of layers traversed at search time. Results
> below are from a single representative run.

## Benchmark Setup

```bash
poetry run python benchmarks/benchmark_pdf_ann_stores.py \
  --pdf The_DynamoDb_Book.pdf \
  --queries-file ./benchmarks/data/pdf_queries.txt \
  --output-csv ./benchmarks/results/pdf_ann_results_3.csv \
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

## Dataset

| Property | Value |
|---|---|
| PDF | *The DynamoDB Book* |
| Records | 881 PDF chunks |
| Queries | 50 semantic queries |
| Embedding model | Sentence Transformer |
| `top_k` | 5 |

## Stores Compared

- `BufferedMatrixInMemVectorStore` — exact search baseline
- `IVFVectorStore`
- `FlatNSWVectorStore`
- `HNSWVectorStore`

## Results

| Store | Parameters | Prepare Time | p95 Search Latency | Avg Recall@5 |
|---|---|---:|---:|---:|
| Exact | `{}` | 0.003 sec | 0.374 ms | 1.000 |
| IVF | `nlist=10, nprobe=1` | 0.143 sec | 0.065 ms | 0.676 |
| IVF | `nlist=10, nprobe=5` | 0.026 sec | 0.151 ms | 0.996 |
| IVF | `nlist=10, nprobe=10` | 0.024 sec | 0.128 ms | 1.000 |
| IVF | `nlist=20, nprobe=1` | 0.036 sec | 0.066 ms | 0.524 |
| IVF | `nlist=20, nprobe=5` | 0.119 sec | 0.121 ms | 0.940 |
| IVF | `nlist=20, nprobe=10` | 0.033 sec | 0.142 ms | 0.992 |
| IVF | `nlist=40, nprobe=1` | 0.064 sec | 0.039 ms | 0.528 |
| IVF | `nlist=40, nprobe=5` | 0.061 sec | 0.065 ms | 0.908 |
| IVF | `nlist=40, nprobe=10` | 0.063 sec | 0.074 ms | 0.972 |
| Flat NSW | `m=8, ef_search=16` | 0.314 sec | 0.116 ms | 0.624 |
| Flat NSW | `m=8, ef_search=32` | 0.308 sec | 0.212 ms | 0.784 |
| Flat NSW | `m=8, ef_search=64` | 0.304 sec | 0.296 ms | 0.892 |
| Flat NSW | `m=16, ef_search=16` | 0.455 sec | 0.183 ms | 0.868 |
| Flat NSW | `m=16, ef_search=32` | 0.455 sec | 0.293 ms | 0.952 |
| Flat NSW | `m=16, ef_search=64` | 0.458 sec | 0.446 ms | 0.996 |
| HNSW | `m=8, ef_construction=32, ef_search=16` | 0.245 sec | 0.148 ms | 0.656 |
| HNSW | `m=8, ef_construction=32, ef_search=32` | 0.246 sec | 0.224 ms | 0.712 |
| HNSW | `m=8, ef_construction=32, ef_search=64` | 0.241 sec | 0.341 ms | 0.820 |
| HNSW | `m=8, ef_construction=64, ef_search=16` | 0.310 sec | 0.147 ms | 0.696 |
| HNSW | `m=8, ef_construction=64, ef_search=32` | 0.307 sec | 0.236 ms | 0.836 |
| HNSW | `m=8, ef_construction=64, ef_search=64` | 0.330 sec | 0.328 ms | 0.912 |
| HNSW | `m=8, ef_construction=128, ef_search=16` | 0.451 sec | 0.177 ms | 0.728 |
| HNSW | `m=8, ef_construction=128, ef_search=32` | 0.440 sec | 0.217 ms | 0.880 |
| HNSW | `m=8, ef_construction=128, ef_search=64` | 0.459 sec | 0.350 ms | 0.892 |
| HNSW | `m=16, ef_construction=32, ef_search=16` | 0.561 sec | 0.287 ms | 0.896 |
| HNSW | `m=16, ef_construction=32, ef_search=32` | 0.575 sec | 0.362 ms | 0.952 |
| HNSW | `m=16, ef_construction=32, ef_search=64` | 0.551 sec | 0.561 ms | 0.984 |
| HNSW | `m=16, ef_construction=64, ef_search=16` | 0.687 sec | 0.250 ms | 0.884 |
| HNSW | `m=16, ef_construction=64, ef_search=32` | 0.660 sec | 0.372 ms | 0.980 |
| HNSW | `m=16, ef_construction=64, ef_search=64` | 0.648 sec | 0.573 ms | 0.992 |
| HNSW | `m=16, ef_construction=128, ef_search=16` | 0.804 sec | 0.264 ms | 0.888 |
| HNSW | `m=16, ef_construction=128, ef_search=32` | 0.775 sec | 0.388 ms | 0.984 |
| HNSW | `m=16, ef_construction=128, ef_search=64` | 0.824 sec | 0.506 ms | 0.996 |

## Search Diagnostics

The diagnostic counters are implementation-specific learning metrics. They are intended
to help understand and compare the internal work performed by each algorithm on the
same query workload. The tables below show averages across 50 queries.

### IVF: Partition Cost is `nlist + vectors_scanned`

| Parameters | Clusters Scanned | Vectors Scanned | Distance Computations | Avg Recall@5 |
|---|---:|---:|---:|---:|
| `nlist=10, nprobe=1` | 1 | 144 | 154 | 0.676 |
| `nlist=10, nprobe=5` | 5 | 554 | 564 | 0.996 |
| `nlist=10, nprobe=10` | 10 | 881 | 891 | 1.000 |
| `nlist=20, nprobe=1` | 1 | 71 | 91 | 0.524 |
| `nlist=20, nprobe=5` | 5 | 318 | 338 | 0.940 |
| `nlist=40, nprobe=1` | 1 | 32 | 72 | 0.528 |
| `nlist=40, nprobe=5` | 5 | 161 | 201 | 0.908 |
| `nlist=40, nprobe=10` | 10 | 289 | 329 | 0.972 |

The formula `distance_computations = nlist + vectors_scanned` holds exactly. The
`nlist` term is the fixed centroid-scoring cost; `vectors_scanned` is the variable
candidate cost controlled by `nprobe`.

`nlist=10, nprobe=10` degenerates to a full scan: all 10 clusters probed, all 881
vectors scored, recall = 1.0.

With `nlist=40, nprobe=5` only 18% of the corpus is scanned (161/881 vectors) and
recall still reaches 0.908 — a direct result of KMeans clustering placing related
vectors in the same partition.

### Flat NSW: `visited_nodes` Always Equals `distance_computations`

| Parameters | Visited Nodes | Graph Hops | Distance Computations | Hop:Dist Ratio | Avg Recall@5 |
|---|---:|---:|---:|---:|---:|
| `m=8, ef_search=16` | 69 | 166 | 69 | 2.4× | 0.624 |
| `m=8, ef_search=32` | 109 | 298 | 109 | 2.7× | 0.784 |
| `m=8, ef_search=64` | 170 | 547 | 170 | 3.2× | 0.892 |
| `m=16, ef_search=16` | 115 | 299 | 115 | 2.6× | 0.868 |
| `m=16, ef_search=32` | 170 | 555 | 170 | 3.3× | 0.952 |
| `m=16, ef_search=64` | 255 | 1058 | 255 | 4.1× | 0.996 |

`visited_nodes` equals `distance_computations` exactly — a structural property of the
implementation. Every node added to the visited set gets exactly one distance
computation; there are no wasted computations from revisits.

The **hop:dist ratio** measures how many edge traversals hit already-visited nodes
and do not produce a new distance computation. This ratio grows with both `m` and
`ef_search`: denser graphs and larger candidate pools increase graph coverage and the
fraction of edges that revisit already-seen nodes.

Higher `m` improves graph quality and recall efficiency. `m=16` at `ef_search=16`
achieves 0.868 recall using 115 distance computations; `m=8` requires `ef_search=64`
and 170 computations to reach a comparable recall band (0.892).

### HNSW: Greedy Upper Layers Add Navigation Overhead

| Parameters | Visited Nodes | Dist Comp | Nav Overhead | Layers | Avg Recall@5 |
|---|---:|---:|---:|---:|---:|
| `m=8, ef_construction=64, ef_search=16` | 63 | 125 | 62 | 7 | 0.696 |
| `m=8, ef_construction=64, ef_search=32` | 104 | 168 | 64 | 7 | 0.836 |
| `m=8, ef_construction=64, ef_search=64` | 168 | 234 | 66 | 8 | 0.912 |
| `m=16, ef_construction=64, ef_search=16` | 106 | 216 | 110 | 7 | 0.884 |
| `m=16, ef_construction=64, ef_search=32` | 167 | 282 | 115 | 7 | 0.980 |
| `m=16, ef_construction=64, ef_search=64` | 255 | 366 | 111 | 6 | 0.992 |

**Navigation overhead** = `distance_computations - visited_nodes`. This is the cost
of greedy descent through upper layers — distance computations that locate a good
layer-0 entry point but are not counted in the `visited_nodes` metric (which tracks
only the layer-0 beam search).

For a given `m`, the navigation overhead stays nearly constant regardless of
`ef_search` (m=8: ~62–66; m=16: ~110–115). This makes sense — `ef_search` only
controls the layer-0 beam search, not the greedy upper-layer traversal.

`layers_traversed` (6–11) varies between runs because the max graph level is set by
the highest randomly-assigned level among all inserted nodes. With
`level_multiplier=1.0`, this follows a geometric distribution and varies between runs.

### HNSW vs Flat NSW at Comparable Recall

At this corpus size (881 vectors), HNSW pays more distance computations than Flat NSW
for roughly comparable recall:

| Recall target | Flat NSW | Dist Comp | HNSW | Dist Comp | HNSW overhead |
|---|---|---:|---|---:|---:|
| ~0.996 | `m=16, ef_search=64` | 255 | `m=16, ef_construction=128, ef_search=64` | 374 | +47% |
| ~0.980 | `m=16, ef_search=32` (0.952) | 170 | `m=16, ef_construction=64, ef_search=32` | 282 | +66% |
| ~0.890 | `m=8, ef_search=64` | 170 | `m=8, ef_construction=64, ef_search=64` | 234 | +38% |

The hierarchical navigation overhead (~60–115 extra distance computations per query)
is not offset by a better entry point at this scale. With only 881 vectors, a random
entry point for Flat NSW is already adequate for the beam search to converge quickly.
HNSW's advantage materialises at scale, where a random entry point into a large graph
would require many more beam search iterations before converging.

## Key Learnings

**IVF:**

```text
lower nprobe  →  lower latency, lower recall
higher nprobe →  higher recall, higher latency
nprobe = nlist degenerates to a full exact scan
```

**Flat NSW:**

```text
higher ef_search  →  better recall, higher latency
higher m          →  denser graph, better recall, higher insert cost
```

**HNSW:**

```text
higher ef_search       →  higher recall, higher latency
higher ef_construction →  better graph quality, slower inserts
higher m               →  denser connectivity, higher recall
```

**Diagnostic learnings:**

IVF's distance computation cost is exactly `nlist + vectors_scanned`. The `nlist`
centroid-scoring is a fixed overhead; `vectors_scanned` is the variable cost controlled
by `nprobe`. Setting `nprobe = nlist` degenerates to a full exact scan.

In Flat NSW, `visited_nodes` always equals `distance_computations`. The hop:distance
ratio (2.4×–4.1×) reflects wasted edge traversals to already-visited nodes, and grows
with both `m` and `ef_search`.

In HNSW, `distance_computations > visited_nodes` always. The gap is the navigation
overhead from greedy upper-layer descent. This gap is approximately constant for a
given `m` and independent of `ef_search`, because upper-layer traversal is controlled
by `m` and the graph topology, not `ef_search`.

At this corpus size, the navigation overhead is net-negative: HNSW uses 38–66% more
distance computations than Flat NSW for comparable recall. The hierarchy only pays off
at scale, where a random entry point to a large graph would cost many more layer-0
beam search iterations to converge.

This benchmark demonstrates three distinct ANN philosophies:

```text
IVF       →  partition then scan  (cost = nlist + cluster_vectors × nprobe)
Flat NSW  →  navigate then refine (cost = visited_nodes, 1:1 with dist_comp)
HNSW      →  hierarchical navigation then refine (cost = nav_overhead + layer_0_beam)
```

## Related Visualizations

See [../visualizations/benchmark_results.md](../visualizations/benchmark_results.md)
for recall–latency and efficiency plots generated from the benchmark CSV.