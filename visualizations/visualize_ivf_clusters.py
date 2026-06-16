#!/usr/bin/env python3
"""
IVF cluster visualization.

Loads a PDF, embeds all chunks with SentenceTransformer, builds an
IVFVectorStore, reduces vectors to 2D via PCA, and plots:
  - every chunk vector coloured by its cluster assignment
  - every cluster centroid
  - a chosen query vector
  - which clusters are probed for that query (highlighted)

Output filename encodes the key parameters so multiple runs don't overwrite
each other (nlist, nprobe, query index).

Usage (from project root):
    poetry run python visualizations/visualize_ivf_clusters.py \\
        --pdf The_DynamoDb_Book.pdf \\
        --queries-file visualizations/queries.txt \\
        --query-index 0 \\
        --nlist 10 \\
        --nprobe 3

    # explicit output path:
    poetry run python visualizations/visualize_ivf_clusters.py \\
        --pdf The_DynamoDb_Book.pdf \\
        --queries-file visualizations/queries.txt \\
        --output-path visualizations/output/ivf_clusters/my_plot.png
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
from vectordb.models import VectorRecord
from vectordb.stores.ivf_inmem import IVFVectorStore

_DEFAULT_OUT = Path(__file__).parent / "output" / "ivf_clusters"


# ── data helpers ──────────────────────────────────────────────────────────────

def load_queries(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def build_store(
    pdf_path: Path,
    nlist: int,
    nprobe: int,
) -> tuple[IVFVectorStore, SentenceTransformerEmbeddingModel]:
    print("Loading PDF chunks...")
    chunks = chunks_from_pdf(pdf_path, RecursiveTextChunker())
    print(f"  {len(chunks)} chunks")

    print("Loading embedding model...")
    model = SentenceTransformerEmbeddingModel()

    print("Embedding chunks (may take a minute)...")
    vectors = model.embed_batch([c.text for c in chunks])

    print("Building IVF index...")
    store = IVFVectorStore(nlist=nlist, nprobe=nprobe)
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        store.insert(VectorRecord(
            id=str(i),
            vector=vec,
            text=chunk.text,
            metadata=chunk.metadata,
        ))
    store.build()
    print(f"  {len(store._ids)} vectors indexed into {nlist} clusters")
    return store, model


def cluster_labels(store: IVFVectorStore) -> np.ndarray:
    """Reconstruct a per-row cluster label array from the inverted lists."""
    labels = np.zeros(len(store._ids), dtype=int)
    for cluster_id, row_indices in store._lists.items():
        for idx in row_indices:
            labels[idx] = cluster_id
    return labels


def probed_cluster_ids(
    store: IVFVectorStore,
    query_norm: np.ndarray,
    nprobe: int,
) -> set[int]:
    """Return the set of cluster IDs that would be probed for this query."""
    scores = store._centroids @ query_norm
    effective = min(nprobe, len(scores))
    top = np.argpartition(-scores, effective - 1)[:effective]
    return set(top.tolist())


# ── plotting ──────────────────────────────────────────────────────────────────

def make_plot(
    store: IVFVectorStore,
    coords_2d: np.ndarray,
    centroids_2d: np.ndarray,
    query_2d: np.ndarray,
    probed_ids: set[int],
    pca: PCA,
    query_text: str,
    nlist: int,
    nprobe: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 9))

    cmap = plt.colormaps["nipy_spectral"].resampled(nlist)
    colors = [cmap(i) for i in range(nlist)]

    # vectors — non-probed clusters (faint)
    for cid in range(nlist):
        if cid in probed_ids:
            continue
        idxs = store._lists.get(cid, [])
        if not idxs:
            continue
        pts = coords_2d[idxs]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=colors[cid], s=14, alpha=0.22,
                   linewidths=0, zorder=1)

    # vectors — probed clusters (vivid)
    for cid in probed_ids:
        idxs = store._lists.get(cid, [])
        if not idxs:
            continue
        pts = coords_2d[idxs]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=colors[cid], s=28, alpha=0.80,
                   edgecolors="white", linewidths=0.4, zorder=2)

    # centroids — non-probed (small diamond, dark edge)
    for cid in range(nlist):
        if cid in probed_ids:
            continue
        cx, cy = centroids_2d[cid]
        ax.scatter(cx, cy, color=colors[cid],
                   marker="D", s=90,
                   edgecolors="#222222", linewidths=1.2, zorder=3)

    # centroids — probed (larger diamond, red edge)
    for cid in probed_ids:
        cx, cy = centroids_2d[cid]
        ax.scatter(cx, cy, color=colors[cid],
                   marker="D", s=220,
                   edgecolors="red", linewidths=2.2, zorder=4)

    # query vector
    qx, qy = query_2d[0]
    ax.scatter(qx, qy, color="red", marker="*", s=500,
               edgecolors="black", linewidths=1.2, zorder=5)

    # query annotation
    label = query_text if len(query_text) <= 80 else query_text[:77] + "..."
    ax.annotate(
        f'Query: "{label}"',
        xy=(qx, qy), xytext=(14, 14), textcoords="offset points",
        fontsize=9, color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
    )

    # axes
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PCA component 1  ({var[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PCA component 2  ({var[1]:.1%} variance)", fontsize=11)
    ax.set_title(
        f"IVF Cluster Visualization — nlist={nlist}, nprobe={nprobe}\n"
        f"2D explained variance: {var[0] + var[1]:.1%}  |  "
        f"{len(store._ids)} vectors  |  {nlist} clusters",
        fontsize=13,
    )

    # legend (proxy artists — one entry per role, not per cluster)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               alpha=0.25, markersize=8, label="Vector (non-probed cluster)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=10, label="Vector (probed cluster)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markeredgecolor="#222222", markersize=9, label="Centroid (non-probed)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markeredgecolor="red", markersize=11, label="Centroid (probed)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=14, label="Query"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize IVF cluster assignments in 2D via PCA"
    )
    parser.add_argument("--pdf", type=Path, required=True,
                        help="PDF file to index")
    parser.add_argument("--queries-file", type=Path, required=True,
                        help="Text file with one query per line")
    parser.add_argument("--query-index", type=int, default=0,
                        help="Which query to highlight (default: 0)")
    parser.add_argument("--nlist", type=int, default=10,
                        help="Number of IVF clusters (default: 10)")
    parser.add_argument("--nprobe", type=int, default=3,
                        help="Number of clusters to probe (default: 3)")
    parser.add_argument("--output-path", type=Path, default=None,
                        help=(
                            "Output PNG path. "
                            "Default: output/ivf_clusters/ivf_clusters_nlist{N}_nprobe{P}_q{Q}.png"
                        ))
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = (
            _DEFAULT_OUT /
            f"ivf_clusters_nlist{args.nlist}_nprobe{args.nprobe}_q{args.query_index}.png"
        )

    # resolve and validate queries
    queries = load_queries(args.queries_file)
    if not queries:
        print(f"Error: no queries found in {args.queries_file}")
        sys.exit(1)
    if args.query_index >= len(queries):
        print(f"Error: --query-index {args.query_index} out of range "
              f"(file has {len(queries)} queries)")
        sys.exit(1)
    query_text = queries[args.query_index]
    print(f"\nQuery [{args.query_index}]: {query_text!r}")
    print(f"nlist={args.nlist}  nprobe={args.nprobe}\n")

    # build
    store, model = build_store(args.pdf, args.nlist, args.nprobe)

    # PCA
    print("\nReducing to 2D with PCA...")
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(store._vectors)
    centroids_2d = pca.transform(store._centroids)
    var = pca.explained_variance_ratio_
    print(f"  PC1={var[0]:.1%}  PC2={var[1]:.1%}  total={var[0]+var[1]:.1%}")

    # embed + project query
    query_vec = model.embed(query_text)
    query_norm = (query_vec / np.linalg.norm(query_vec)).astype(np.float32)
    query_2d = pca.transform(query_norm.reshape(1, -1))

    # probed clusters
    probed = probed_cluster_ids(store, query_norm, args.nprobe)
    print(f"  Probed cluster IDs: {sorted(probed)}\n")

    make_plot(
        store=store,
        coords_2d=coords_2d,
        centroids_2d=centroids_2d,
        query_2d=query_2d,
        probed_ids=probed,
        pca=pca,
        query_text=query_text,
        nlist=args.nlist,
        nprobe=args.nprobe,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
