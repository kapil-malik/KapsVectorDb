#!/usr/bin/env python3
"""
Visualize ANN benchmark results from pdf_ann_results_3.csv.

Produces eight PNG plots in the output/ directory:

  Core scatter plots (spec):
  1. recall_vs_latency.png               — p95 latency (x) vs recall_avg (y), all stores
  2. recall_vs_distance_computations.png — dist computations (x) vs recall_avg (y), ANN stores
  3. recall_vs_visited_nodes.png         — visited nodes (x) vs recall_avg (y), graph stores
  4. latency_vs_vectors_scanned.png      — vectors scanned (x) vs p95 latency (y), IVF

  Parameter sweep plots (extra):
  5. ivf_parameter_sweep.png             — IVF: nprobe vs recall/latency for each nlist
  6. flat_nsw_parameter_sweep.png        — FlatNSW: ef_search vs recall/latency for each m
  7. hnsw_parameter_sweep.png            — HNSW: ef_search vs recall/latency per (m, ef_construction)
  8. graph_store_comparison.png          — FlatNSW vs HNSW distance-computation efficiency at matched m

Usage:
    poetry run python visualizations/visualize_ann_benchmark_results.py
    poetry run python visualizations/visualize_ann_benchmark_results.py \\
        --input-csv benchmarks/results/pdf_ann_results_3.csv \\
        --output-dir visualizations/output/benchmark
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

STORE_STYLE: dict[str, dict] = {
    "exact":    {"color": "#888888", "marker": "D", "label": "Exact"},
    "ivf":      {"color": "#1565C0", "marker": "o", "label": "IVF"},
    "flat_nsw": {"color": "#E65100", "marker": "s", "label": "FlatNSW"},
    "hnsw":     {"color": "#6A1B9A", "marker": "^", "label": "HNSW"},
}


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["params"] = df["store_parameters"].apply(json.loads)
    return df


def _has_columns(df: pd.DataFrame, *cols: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  skipped — missing columns: {missing}")
        return False
    return True


def _savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 1: p95 latency vs recall (all stores) ─────────────────────────────

def plot_recall_vs_latency(df: pd.DataFrame, out_dir: Path) -> None:
    if not _has_columns(df, "search_p95_latency_ms", "recall_avg"):
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for store, style in STORE_STYLE.items():
        rows = df[df["store"] == store]
        if rows.empty:
            continue
        ax.scatter(
            rows["search_p95_latency_ms"],
            rows["recall_avg"],
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_xlabel("p95 search latency (ms)", fontsize=12)
    ax.set_ylabel("Recall@10 (avg)", fontsize=12)
    ax.set_title("ANN Stores — p95 Latency vs Recall", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0.45, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)

    _savefig(fig, out_dir, "recall_vs_latency.png")


# ── Figure 2: distance computations vs recall (ANN stores) ───────────────────

def plot_recall_vs_distance_computations(df: pd.DataFrame, out_dir: Path) -> None:
    if not _has_columns(df, "diag_avg_distance_computations", "recall_avg"):
        return

    ann = df[df["store"].isin(["ivf", "flat_nsw", "hnsw"])]

    fig, ax = plt.subplots(figsize=(9, 6))

    for store, style in STORE_STYLE.items():
        rows = ann[ann["store"] == store]
        if rows.empty:
            continue
        ax.scatter(
            rows["diag_avg_distance_computations"],
            rows["recall_avg"],
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    total = int(df["records"].iloc[0])
    ax.axvline(
        total,
        color="#888888",
        linestyle=":",
        linewidth=1.2,
        label=f"Full brute-force scan ({total})",
    )

    ax.set_xlabel("Avg distance computations per query", fontsize=12)
    ax.set_ylabel("Recall@10 (avg)", fontsize=12)
    ax.set_title("ANN Stores — Distance Computations vs Recall", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0.45, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)

    _savefig(fig, out_dir, "recall_vs_distance_computations.png")


# ── Figure 3: visited nodes vs recall (graph stores) ─────────────────────────

def plot_recall_vs_visited_nodes(df: pd.DataFrame, out_dir: Path) -> None:
    if not _has_columns(df, "diag_avg_visited_nodes", "recall_avg"):
        return

    # Only graph stores track visited nodes; filter out rows where it is 0.
    graph = df[df["diag_avg_visited_nodes"] > 0]
    if graph.empty:
        print("  skipped recall_vs_visited_nodes.png — no rows with visited_nodes > 0")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for store, style in STORE_STYLE.items():
        rows = graph[graph["store"] == store]
        if rows.empty:
            continue
        ax.scatter(
            rows["diag_avg_visited_nodes"],
            rows["recall_avg"],
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_xlabel("Avg visited nodes per query", fontsize=12)
    ax.set_ylabel("Recall@10 (avg)", fontsize=12)
    ax.set_title("Graph Stores — Visited Nodes vs Recall", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0.45, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)

    _savefig(fig, out_dir, "recall_vs_visited_nodes.png")


# ── Figure 4: vectors scanned vs p95 latency (IVF) ───────────────────────────

def plot_latency_vs_vectors_scanned(df: pd.DataFrame, out_dir: Path) -> None:
    if not _has_columns(df, "diag_avg_vectors_scanned", "search_p95_latency_ms"):
        return

    # Only IVF tracks vectors_scanned; filter out rows where it is 0.
    ivf = df[df["diag_avg_vectors_scanned"] > 0]
    if ivf.empty:
        print("  skipped latency_vs_vectors_scanned.png — no rows with vectors_scanned > 0")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for store, style in STORE_STYLE.items():
        rows = ivf[ivf["store"] == store]
        if rows.empty:
            continue
        ax.scatter(
            rows["diag_avg_vectors_scanned"],
            rows["search_p95_latency_ms"],
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_xlabel("Avg vectors scanned per query", fontsize=12)
    ax.set_ylabel("p95 search latency (ms)", fontsize=12)
    ax.set_title("IVF — Vectors Scanned vs p95 Latency", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    _savefig(fig, out_dir, "latency_vs_vectors_scanned.png")


# ── Figure 5: IVF parameter sweep ────────────────────────────────────────────

def plot_ivf_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    ivf = df[df["store"] == "ivf"].copy()
    if ivf.empty:
        return
    ivf["nlist"] = ivf["params"].apply(lambda p: p["nlist"])
    ivf["nprobe"] = ivf["params"].apply(lambda p: p["nprobe"])

    palette = {10: "#0D47A1", 20: "#1976D2", 40: "#64B5F6"}

    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(12, 5))

    for nlist, grp in ivf.groupby("nlist"):
        grp = grp.sort_values("nprobe")
        color = palette.get(int(nlist), "gray")
        label = f"nlist={nlist}"
        ax_r.plot(grp["nprobe"], grp["recall_avg"], marker="o", color=color, label=label)
        ax_l.plot(grp["nprobe"], grp["search_p95_latency_ms"], marker="o", color=color, label=label)

    for ax in (ax_r, ax_l):
        ax.set_xlabel("nprobe", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax_r.set_ylabel("Recall@10 (avg)", fontsize=11)
    ax_r.set_ylim(0, 1.05)
    ax_r.set_title("IVF — nprobe vs Recall", fontsize=12)

    ax_l.set_ylabel("p95 latency (ms)", fontsize=11)
    ax_l.set_title("IVF — nprobe vs p95 Latency", fontsize=12)

    fig.suptitle("IVF Parameter Sweep", fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig(fig, out_dir, "ivf_parameter_sweep.png")


# ── Figure 6: FlatNSW parameter sweep ────────────────────────────────────────

def plot_flat_nsw_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    nsw = df[df["store"] == "flat_nsw"].copy()
    if nsw.empty:
        return
    nsw["m"] = nsw["params"].apply(lambda p: p["m"])
    nsw["ef_search"] = nsw["params"].apply(lambda p: p["ef_search"])

    palette = {8: "#BF360C", 16: "#FF8F00"}

    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(12, 5))

    for m_val, grp in nsw.groupby("m"):
        grp = grp.sort_values("ef_search")
        color = palette.get(int(m_val), "gray")
        label = f"m={m_val}"
        ax_r.plot(grp["ef_search"], grp["recall_avg"], marker="s", color=color, label=label)
        ax_l.plot(grp["ef_search"], grp["search_p95_latency_ms"], marker="s", color=color, label=label)

    for ax in (ax_r, ax_l):
        ax.set_xlabel("ef_search", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax_r.set_ylabel("Recall@10 (avg)", fontsize=11)
    ax_r.set_ylim(0, 1.05)
    ax_r.set_title("FlatNSW — ef_search vs Recall", fontsize=12)

    ax_l.set_ylabel("p95 latency (ms)", fontsize=11)
    ax_l.set_title("FlatNSW — ef_search vs p95 Latency", fontsize=12)

    fig.suptitle("FlatNSW Parameter Sweep", fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig(fig, out_dir, "flat_nsw_parameter_sweep.png")


# ── Figure 7: HNSW parameter sweep ───────────────────────────────────────────

def plot_hnsw_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    hnsw = df[df["store"] == "hnsw"].copy()
    if hnsw.empty:
        return
    hnsw["m"] = hnsw["params"].apply(lambda p: p["m"])
    hnsw["ef_construction"] = hnsw["params"].apply(lambda p: p["ef_construction"])
    hnsw["ef_search"] = hnsw["params"].apply(lambda p: p["ef_search"])

    m_values = sorted(hnsw["m"].unique())
    efc_palette = {32: "#4A148C", 64: "#7B1FA2", 128: "#CE93D8"}

    fig, axes = plt.subplots(len(m_values), 2, figsize=(13, 5 * len(m_values)))

    for row, m_val in enumerate(m_values):
        ax_r, ax_l = axes[row][0], axes[row][1]
        grp_m = hnsw[hnsw["m"] == m_val]

        for efc, grp in grp_m.groupby("ef_construction"):
            grp = grp.sort_values("ef_search")
            color = efc_palette.get(int(efc), "gray")
            label = f"ef_construction={efc}"
            ax_r.plot(grp["ef_search"], grp["recall_avg"], marker="^", color=color, label=label)
            ax_l.plot(grp["ef_search"], grp["search_p95_latency_ms"], marker="^", color=color, label=label)

        for ax in (ax_r, ax_l):
            ax.set_xlabel("ef_search", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax_r.set_ylabel("Recall@10 (avg)", fontsize=10)
        ax_r.set_ylim(0, 1.05)
        ax_r.set_title(f"HNSW m={m_val} — ef_search vs Recall", fontsize=11)

        ax_l.set_ylabel("p95 latency (ms)", fontsize=10)
        ax_l.set_title(f"HNSW m={m_val} — ef_search vs p95 Latency", fontsize=11)

    fig.suptitle("HNSW Parameter Sweep", fontsize=13)
    fig.tight_layout()
    _savefig(fig, out_dir, "hnsw_parameter_sweep.png")


# ── Figure 8: FlatNSW vs HNSW efficiency (matched m) ────────────────────────

def plot_graph_store_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    if not _has_columns(df, "diag_avg_distance_computations"):
        return

    nsw = df[df["store"] == "flat_nsw"].copy()
    hnsw = df[df["store"] == "hnsw"].copy()
    if nsw.empty or hnsw.empty:
        return
    nsw["m"] = nsw["params"].apply(lambda p: p["m"])
    hnsw["m"] = hnsw["params"].apply(lambda p: p["m"])

    m_values = sorted(nsw["m"].unique())
    fig, axes = plt.subplots(1, len(m_values), figsize=(6 * len(m_values), 5), sharey=True)

    for ax, m_val in zip(axes, m_values):
        nsw_m = nsw[nsw["m"] == m_val]
        hnsw_m = hnsw[hnsw["m"] == m_val]

        ax.scatter(
            nsw_m["diag_avg_distance_computations"],
            nsw_m["recall_avg"],
            color=STORE_STYLE["flat_nsw"]["color"],
            marker=STORE_STYLE["flat_nsw"]["marker"],
            label="FlatNSW",
            s=80,
            alpha=0.85,
            zorder=3,
        )
        ax.scatter(
            hnsw_m["diag_avg_distance_computations"],
            hnsw_m["recall_avg"],
            color=STORE_STYLE["hnsw"]["color"],
            marker=STORE_STYLE["hnsw"]["marker"],
            label="HNSW",
            s=80,
            alpha=0.85,
            zorder=3,
        )
        ax.set_xlabel("Avg distance computations per query", fontsize=11)
        ax.set_title(f"m={m_val}: Distance Computations vs Recall", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Recall@10 (avg)", fontsize=11)
    fig.suptitle("FlatNSW vs HNSW — Search Efficiency at Matched m", fontsize=13)
    fig.tight_layout()
    _savefig(fig, out_dir, "graph_store_comparison.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    default_csv = (
        Path(__file__).parent.parent / "benchmarks" / "results" / "pdf_ann_results_3.csv"
    )
    default_out = Path(__file__).parent / "output" / "benchmark"

    parser = argparse.ArgumentParser(description="Visualize ANN benchmark results")
    parser.add_argument(
        "--input-csv", type=Path, default=default_csv,
        help="Path to benchmark CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_out,
        help="Directory for output PNGs (default: %(default)s)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input_csv)
    print(f"Loaded {len(df)} rows from {args.input_csv}")
    print(f"Writing plots to {args.output_dir}/\n")

    plot_recall_vs_latency(df, args.output_dir)
    plot_recall_vs_distance_computations(df, args.output_dir)
    plot_recall_vs_visited_nodes(df, args.output_dir)
    plot_latency_vs_vectors_scanned(df, args.output_dir)
    plot_ivf_sweep(df, args.output_dir)
    plot_flat_nsw_sweep(df, args.output_dir)
    plot_hnsw_sweep(df, args.output_dir)
    plot_graph_store_comparison(df, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
