# Graph Traversal Visualization

## Goal

Show which nodes and edges a graph-based ANN search actually visits when answering a query.
The visualization answers:

- How much of the graph does a single search explore? (visited fraction)
- Are the top-k results reachable from the entry point via a tight traversal path?
- How does HNSW layer structure affect traversal density — is the base layer (level 0)
  much denser than upper layers?

All vectors are reduced to 2D via PCA for plotting. Sentence embeddings live in 384 dimensions,
so the 2D projection retains only ~20% of total variance. Cluster overlap in 2D does not imply
poor separation in the original space.

## Script

```
visualizations/visualize_graph_traversal.py
```

## Input

| Input | Description |
|---|---|
| `--store` | `flat_nsw` or `hnsw` |
| PDF file | Any PDF to index (`--pdf`) |
| Queries file | Plain text, one query per line (`--queries-file`) |
| `--query-index` | Which query to highlight (0-indexed) |
| `--top-k` | Number of results (default: 5) |
| `--max-chunks` | Limit PDF chunks indexed (default: 200; keeps graph readable) |
| `--m` | Graph degree parameter M (default: 8) |
| `--ef-search` | Candidate pool size at search time (default: 32) |
| `--ef-construction` | HNSW only — candidate pool size at build time (default: 64) |
| `--level` | HNSW only — which layer to draw edges and nodes for (default: 0) |

## Run

```bash
# FlatNSW
poetry run python visualizations/visualize_graph_traversal.py \
  --store flat_nsw \
  --pdf The_DynamoDb_Book.pdf \
  --queries-file visualizations/queries.txt \
  --query-index 0 \
  --max-chunks 200

# HNSW base layer (level 0)
poetry run python visualizations/visualize_graph_traversal.py \
  --store hnsw \
  --pdf The_DynamoDb_Book.pdf \
  --queries-file visualizations/queries.txt \
  --query-index 0 \
  --max-chunks 200 \
  --level 0

# HNSW upper layer (level 1 — sparser, shows greedy-descent structure)
poetry run python visualizations/visualize_graph_traversal.py \
  --store hnsw \
  --pdf The_DynamoDb_Book.pdf \
  --queries-file visualizations/queries.txt \
  --query-index 0 \
  --max-chunks 200 \
  --level 1
```

Output filenames encode key parameters so multiple runs coexist:

```
graph_{store}_m{M}_ef{ef_search}_chunks{N}[_lvl{L}]_q{Q}.png
```

To override:

```bash
poetry run python visualizations/visualize_graph_traversal.py \
  --store hnsw \
  --pdf The_DynamoDb_Book.pdf \
  --queries-file visualizations/queries.txt \
  --output-path visualizations/output/graph_traversal/custom_name.png
```

## Visual Legend

| Element | Appearance | Meaning |
|---|---|---|
| Small gray dots | Light gray, low alpha | Nodes not visited during this search |
| Larger orange dots | Orange, medium alpha | Nodes visited (explored) during search |
| Larger green dots | Green, white edge | Top-k result nodes |
| Blue diamond `◆` | Blue, white edge | Entry point for the search at the selected layer (for HNSW, this is the level-L entry after untracked greedy descent, not the global top-layer entry) |
| Red star `★` | Red, black edge | Query vector (projected to 2D) |
| Faint gray lines | Very low alpha | All graph edges at this layer |
| Orange lines | Medium alpha | Traversal edges (both endpoints visited) |

**How traversal is tracked:** An opt-in field `visited_node_ids: list[str] | None` was added to
`SearchDiagnostics`. Normal search paths leave it `None` (zero overhead). The visualization
script opts in by passing `SearchDiagnostics(visited_node_ids=[])` to the internal helper
methods, which append each node ID as it enters the visited set.

For HNSW, tracking is **level-isolated**: the script descends from the top layer to the target
level without recording any nodes, then tracks only the search phase at the selected level.
This means orange nodes on a level-L plot are exclusively the nodes scored during that layer's
search — upper-layer greedy descent and base-layer beam activity on other levels are excluded.

---

## Sample Outputs

### FlatNSW — Query: "what is a partition key and how does DynamoDB use it?"

```
--store flat_nsw --max-chunks 200 --m 8 --ef-search 32 --query-index 0
200 nodes · 800 edges · 77 nodes visited (38.5%)
```

![FlatNSW graph traversal](../../visualizations/output/graph_traversal/graph_flat_nsw_m8_ef32_chunks200_q0.png)

FlatNSW is a single-layer graph. The search starts at the entry point (blue diamond) and
expands outward via the beam search (ef_search=32). Orange nodes show the explored region;
traversal edges connect visited neighbors. The 5 top-k results (green) sit within the visited
subgraph — the search reached them via the graph connectivity. 38.5% of all nodes were visited,
reflecting the ef_search=32 candidate pool limiting expansion after the beam fills.

---

### HNSW Level 0 — Query: "what is a partition key and how does DynamoDB use it?"

```
--store hnsw --max-chunks 200 --m 8 --ef-search 32 --level 0 --query-index 0
200 nodes · 1180 edges · 83 unique nodes visited
```

![HNSW level 0 graph traversal](../../visualizations/output/graph_traversal/graph_hnsw_m8_ef32_chunks200_lvl0_q0.png)

HNSW level 0 is the densest layer — every node participates and has up to M=8 bidirectional
links. The search first descends greedily through upper layers to find a good entry candidate
(untracked). The blue diamond marks where the level-0 beam search actually starts — the node
handed off by the untracked greedy descent. From there, the ef_search=32 beam expands outward;
all nodes it visits are shown in orange, with orange edges connecting visited pairs. The visited
count (83) is comparable to FlatNSW (77) because both now reflect exclusively the base-layer
beam, making the comparison fair.

---

### HNSW Level 1 — Query: "what is a partition key and how does DynamoDB use it?"

```
--store hnsw --max-chunks 200 --m 8 --ef-search 32 --level 1 --query-index 0
72 nodes at level 1 · 437 edges · 13 unique nodes visited
```

![HNSW level 1 graph traversal](../../visualizations/output/graph_traversal/graph_hnsw_m8_ef32_chunks200_lvl1_q0.png)

Level 1 is sparser — only ~36% of the 200 nodes are promoted here, and edges show the
long-range shortcuts that make upper-layer navigation fast. The blue diamond marks the actual
entry into the level-1 greedy search — the node handed off by the untracked descent from higher
layers. From it, the greedy walk follows the single best neighbor at each step, visiting only
13 nodes before converging. Orange edges connect visited pairs within the level-1 graph,
so the traversal path is fully visible and correctly rooted at the blue diamond. This accurately
captures how upper layers behave: a tight greedy walk that quickly narrows the search region
before handing off to the denser base layer.
