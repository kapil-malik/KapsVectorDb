#!/usr/bin/env python3
"""
Graph traversal visualization for FlatNSW and HNSW.

Loads a PDF, embeds chunks, builds the chosen graph store, reduces all
vectors to 2D via PCA, and plots:
  - all graph edges at the selected layer (faint gray)
  - edges between nodes visited during the search (highlighted orange)
  - all nodes (gray), visited nodes (orange), top-k results (green)
  - the entry point (blue diamond) and query vector (red star)

Usage (from project root):
    poetry run python visualizations/visualize_graph_traversal.py \
        --store flat_nsw \
        --pdf The_DynamoDb_Book.pdf \
        --queries-file visualizations/queries.txt \
        --query-index 0 \
        --max-chunks 200

    # HNSW at base layer (default):
    poetry run python visualizations/visualize_graph_traversal.py \
        --store hnsw \
        --pdf The_DynamoDb_Book.pdf \
        --queries-file visualizations/queries.txt \
        --query-index 0 \
        --max-chunks 200

    # HNSW upper layer (sparser, shows greedy path):
    poetry run python visualizations/visualize_graph_traversal.py \
        --store hnsw \
        --pdf The_DynamoDb_Book.pdf \
        --queries-file visualizations/queries.txt \
        --query-index 0 \
        --max-chunks 200 \
        --level 1
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from vectordb.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from vectordb.ingestion.chunker import RecursiveTextChunker
from vectordb.ingestion.pdf_ingestion import chunks_from_pdf
from vectordb.models import SearchDiagnostics, VectorRecord
from vectordb.stores.flat_nsw_inmem import FlatNSWVectorStore
from vectordb.stores.hnsw_inmem import HNSWVectorStore

_DEFAULT_OUT = Path(__file__).parent / "output" / "graph_traversal"


# ── data helpers ──────────────────────────────────────────────────────────────

def load_queries(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def embed_chunks(
    pdf_path: Path,
    max_chunks: int,
) -> tuple[list, list[np.ndarray], SentenceTransformerEmbeddingModel]:
    print("Loading PDF chunks...")
    chunks = chunks_from_pdf(pdf_path, RecursiveTextChunker())
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
    print(f"  Using {len(chunks)} chunks (--max-chunks={max_chunks})")

    print("Loading embedding model...")
    model = SentenceTransformerEmbeddingModel()

    print("Embedding chunks...")
    vectors = model.embed_batch([c.text for c in chunks])
    return chunks, vectors, model


def build_flat_nsw(
    chunks,
    vectors,
    m: int,
    ef_search: int,
) -> FlatNSWVectorStore:
    print(f"Building FlatNSW (m={m}, ef_search={ef_search})...")
    store = FlatNSWVectorStore(m=m, ef_search=ef_search)
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        store.insert(VectorRecord(
            id=str(i),
            vector=vec,
            text=chunk.text,
            metadata=chunk.metadata,
        ))
    edge_count = sum(len(v) for v in store._neighbors.values()) // 2
    print(f"  {store.count()} nodes, {edge_count} edges")
    return store


def build_hnsw(
    chunks,
    vectors,
    m: int,
    ef_construction: int,
    ef_search: int,
) -> HNSWVectorStore:
    print(f"Building HNSW (m={m}, ef_construction={ef_construction}, ef_search={ef_search})...")
    store = HNSWVectorStore(m=m, ef_construction=ef_construction, ef_search=ef_search)
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        store.insert(VectorRecord(
            id=str(i),
            vector=vec,
            text=chunk.text,
            metadata=chunk.metadata,
        ))
    n_level0 = len(store._neighbors.get(0, {}))
    print(f"  {store.count()} nodes, max_level={store._max_level}, nodes at level 0={n_level0}")
    return store


# ── traversal ─────────────────────────────────────────────────────────────────

def traverse_flat_nsw(
    store: FlatNSWVectorStore,
    query_norm: np.ndarray,
    top_k: int,
) -> tuple[list[str], list[tuple[float, str]]]:
    diag = SearchDiagnostics(visited_node_ids=[])
    best_candidates = store._search_with_entry_point(query_norm, diag)
    best_candidates.sort(key=lambda x: x[0], reverse=True)
    results = [
        (s, rid) for s, rid in best_candidates
        if rid not in store._tombstone_ids
    ][:top_k]
    return diag.visited_node_ids or [], results


def traverse_hnsw(
    store: HNSWVectorStore,
    query_norm: np.ndarray,
    top_k: int,
    level: int,
) -> tuple[list[str], list[tuple[float, str]], str | None]:
    # Navigate to the target level without tracking (untracked descent)
    current = store._entry_point_id
    for lv in range(store._max_level, level, -1):
        current = store._greedy_search_layer(query_norm, current, lv)

    level_entry_id = current  # actual entry point into the level-L search

    # Track only the search phase at the selected level
    diag = SearchDiagnostics(visited_node_ids=[])
    if level == 0:
        best = store._search_layer(query_norm, current, 0, store.ef_search, diag)
        best.sort(key=lambda x: x[0], reverse=True)
        results = [
            (s, rid) for s, rid in best
            if rid not in store._tombstone_ids
        ][:top_k]
    else:
        # Track greedy search at exactly this level
        store._greedy_search_layer(query_norm, current, level, diag)
        # Get actual top-k results via separate untracked full search
        full_current = store._entry_point_id
        for lv in range(store._max_level, 0, -1):
            full_current = store._greedy_search_layer(query_norm, full_current, lv)
        best = store._search_layer(query_norm, full_current, 0, store.ef_search)
        best.sort(key=lambda x: x[0], reverse=True)
        results = [
            (s, rid) for s, rid in best
            if rid not in store._tombstone_ids
        ][:top_k]

    return diag.visited_node_ids or [], results, level_entry_id


# ── graph structure helpers ───────────────────────────────────────────────────

def flat_nsw_graph(store: FlatNSWVectorStore):
    """Return (node_ids, undirected_edges) for FlatNSW (single layer)."""
    node_ids = list(store._vectors.keys())
    edges = []
    seen: set[tuple[str, str]] = set()
    for u, neighbors in store._neighbors.items():
        for v in neighbors:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                edges.append(key)
    return node_ids, edges


def hnsw_graph(store: HNSWVectorStore, level: int):
    """Return (node_ids, undirected_edges) for HNSW at a given layer."""
    neighbors_at_level = store._neighbors.get(level, {})
    node_ids = [nid for nid, lv in store._levels.items() if lv >= level]
    edges = []
    seen: set[tuple[str, str]] = set()
    for u, neighbors in neighbors_at_level.items():
        for v in neighbors:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                edges.append(key)
    return node_ids, edges


# ── plotting ──────────────────────────────────────────────────────────────────

def make_plot(
    node_ids: list[str],
    edges: list[tuple[str, str]],
    visited_ids: list[str],
    result_ids: set[str],
    entry_point_id: str | None,
    coords_2d: np.ndarray,
    id_to_idx: dict[str, int],
    query_2d: np.ndarray,
    query_text: str,
    pca: PCA,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))
    visited_set = set(visited_ids)
    node_set = set(node_ids)

    # ── edges ────────────────────────────────────────────────────────────────
    # All edges — very faint
    for u, v in edges:
        if u not in id_to_idx or v not in id_to_idx:
            continue
        xu, yu = coords_2d[id_to_idx[u]]
        xv, yv = coords_2d[id_to_idx[v]]
        ax.plot([xu, xv], [yu, yv], color="#bbbbbb", alpha=0.07, lw=0.5, zorder=1)

    # Traversal edges — both endpoints visited
    for u, v in edges:
        if u in visited_set and v in visited_set:
            if u not in id_to_idx or v not in id_to_idx:
                continue
            xu, yu = coords_2d[id_to_idx[u]]
            xv, yv = coords_2d[id_to_idx[v]]
            ax.plot([xu, xv], [yu, yv], color="#FF8C00", alpha=0.45, lw=1.3, zorder=2)

    # ── nodes ────────────────────────────────────────────────────────────────
    # Non-visited, non-result nodes (small, gray)
    non_visited = [
        nid for nid in node_ids
        if nid not in visited_set
        and nid not in result_ids
        and nid != entry_point_id
    ]
    nv_idxs = [id_to_idx[nid] for nid in non_visited if nid in id_to_idx]
    if nv_idxs:
        pts = coords_2d[nv_idxs]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color="#aaaaaa", s=12, alpha=0.35, linewidths=0, zorder=3)

    # Visited nodes (not results, not entry point)
    visited_plain = [
        nid for nid in visited_set
        if nid not in result_ids
        and nid != entry_point_id
        and nid in node_set
    ]
    vp_idxs = [id_to_idx[nid] for nid in visited_plain if nid in id_to_idx]
    if vp_idxs:
        pts = coords_2d[vp_idxs]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color="#FF8C00", s=38, alpha=0.88,
                   edgecolors="white", linewidths=0.5, zorder=4,
                   label=f"Visited ({len(vp_idxs)} nodes)")

    # Result nodes
    r_idxs = [id_to_idx[nid] for nid in result_ids if nid in id_to_idx]
    if r_idxs:
        pts = coords_2d[r_idxs]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color="#2ecc71", s=95,
                   edgecolors="white", linewidths=1.2, zorder=5,
                   label=f"Top-k results ({len(r_idxs)})")

    # Entry point
    if entry_point_id and entry_point_id in id_to_idx:
        ex, ey = coords_2d[id_to_idx[entry_point_id]]
        ax.scatter(ex, ey, color="#3498db", marker="D", s=130,
                   edgecolors="white", linewidths=1.2, zorder=5,
                   label="Entry point")

    # Query vector
    qx, qy = query_2d[0]
    ax.scatter(qx, qy, color="red", marker="*", s=520,
               edgecolors="black", linewidths=1.1, zorder=6,
               label="Query")

    label = query_text if len(query_text) <= 80 else query_text[:77] + "..."
    ax.annotate(
        f'Query: "{label}"',
        xy=(qx, qy), xytext=(14, 14), textcoords="offset points",
        fontsize=9, color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
    )

    # ── legend & axes ─────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               alpha=0.4, markersize=7, label=f"Nodes not visited ({len(non_visited)})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF8C00",
               markersize=9, label=f"Visited ({len(vp_idxs)} nodes)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markersize=11, label=f"Top-k results ({len(r_idxs)})"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#3498db",
               markersize=9, label="Entry point"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=14, label="Query"),
        Line2D([0], [0], color="#bbbbbb", alpha=0.5, lw=1, label="Graph edge"),
        Line2D([0], [0], color="#FF8C00", alpha=0.8, lw=1.5, label="Traversal edge"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PCA component 1  ({var[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PCA component 2  ({var[1]:.1%} variance)", fontsize=11)
    ax.set_title(
        f"{title}\n"
        f"2D explained variance: {var[0] + var[1]:.1%}  |  "
        f"{len(node_ids)} nodes  |  {len(edges)} edges  |  {len(visited_set)} visited",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize graph traversal for FlatNSW or HNSW in 2D via PCA"
    )
    parser.add_argument(
        "--store", choices=["flat_nsw", "hnsw"], required=True,
        help="Which store to build and visualize",
    )
    parser.add_argument(
        "--pdf", type=Path, required=True,
        help="PDF file to index",
    )
    parser.add_argument(
        "--queries-file", type=Path, required=True,
        help="Text file with one query per line",
    )
    parser.add_argument(
        "--query-index", type=int, default=0,
        help="Which query to highlight (default: 0)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=200,
        help="Maximum number of PDF chunks to index (default: 200)",
    )
    parser.add_argument(
        "--m", type=int, default=8,
        help="Graph degree parameter M (default: 8)",
    )
    parser.add_argument(
        "--ef-search", type=int, default=32,
        help="ef_search for base-layer search (default: 32)",
    )
    parser.add_argument(
        "--ef-construction", type=int, default=64,
        help="ef_construction (HNSW only, default: 64)",
    )
    parser.add_argument(
        "--level", type=int, default=0,
        help="Which HNSW layer to visualize (default: 0; ignored for flat_nsw)",
    )
    parser.add_argument(
        "--output-path", type=Path, default=None,
        help="Output PNG path (default: output/graph_traversal/<auto>.png)",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries_file)
    if not queries:
        print(f"Error: no queries found in {args.queries_file}")
        sys.exit(1)
    if args.query_index >= len(queries):
        print(
            f"Error: --query-index {args.query_index} out of range "
            f"(file has {len(queries)} queries)"
        )
        sys.exit(1)
    query_text = queries[args.query_index]

    if args.output_path is None:
        stem = (
            f"graph_{args.store}"
            f"_m{args.m}"
            f"_ef{args.ef_search}"
            f"_chunks{args.max_chunks}"
            + (f"_lvl{args.level}" if args.store == "hnsw" else "")
            + f"_q{args.query_index}.png"
        )
        args.output_path = _DEFAULT_OUT / stem

    print(f"\nQuery [{args.query_index}]: {query_text!r}")

    chunks, vectors, model = embed_chunks(args.pdf, args.max_chunks)

    # Build store
    if args.store == "flat_nsw":
        store = build_flat_nsw(chunks, vectors, args.m, args.ef_search)
        node_ids, edges = flat_nsw_graph(store)
        entry_point_id = store._entry_point_id
        level_label = ""
    else:
        store = build_hnsw(chunks, vectors, args.m, args.ef_construction, args.ef_search)
        if args.level > store._max_level:
            print(f"Warning: --level {args.level} > max_level={store._max_level}; using level 0")
            args.level = 0
        node_ids, edges = hnsw_graph(store, args.level)
        entry_point_id = store._entry_point_id
        level_label = f" level={args.level}"
        print(f"  Nodes at level {args.level}: {len(node_ids)}, edges: {len(edges)}")

    # PCA on all stored vectors (dict-based storage)
    print("\nReducing to 2D with PCA...")
    all_ids = list(store._vectors.keys())
    matrix = np.stack([store._vectors[nid] for nid in all_ids])
    pca = PCA(n_components=2)
    all_coords = pca.fit_transform(matrix)
    id_to_idx = {nid: i for i, nid in enumerate(all_ids)}
    var = pca.explained_variance_ratio_
    print(f"  PC1={var[0]:.1%}  PC2={var[1]:.1%}  total={var[0]+var[1]:.1%}")

    # Embed and project query
    query_vec = model.embed(query_text)
    query_norm = (query_vec / np.linalg.norm(query_vec)).astype(np.float32)
    query_2d = pca.transform(query_norm.reshape(1, -1))

    # Traverse
    print("\nRunning search with traversal tracking...")
    if args.store == "flat_nsw":
        visited_ids, results = traverse_flat_nsw(store, query_norm, args.top_k)
    else:
        visited_ids, results, entry_point_id = traverse_hnsw(
            store, query_norm, args.top_k, args.level
        )

    result_ids = {rid for _, rid in results}
    print(f"  Visited: {len(set(visited_ids))} unique nodes")
    print(f"  Results: {[rid for _, rid in results]}")

    title = (
        f"Graph Traversal — {args.store.upper()}"
        + level_label
        + f"  |  m={args.m}  ef_search={args.ef_search}  chunks={len(node_ids)}"
    )

    make_plot(
        node_ids=node_ids,
        edges=edges,
        visited_ids=visited_ids,
        result_ids=result_ids,
        entry_point_id=entry_point_id,
        coords_2d=all_coords,
        id_to_idx=id_to_idx,
        query_2d=query_2d,
        query_text=query_text,
        pca=pca,
        title=title,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
