#!/usr/bin/env python3
"""
Visualize how search latency scales with record count across all exact vector stores.

Consumes: benchmarks/results/store_latency_scaling.csv

Produces two PNG plots in the output directory:
  1. p95_latency_vs_records.png           — all 6 stores on a log-log chart
  2. p95_latency_vs_records_fast_stores.png — matrix-backed stores only (zoomed)

Usage:
    poetry run python visualizations/visualize_store_latency_scaling.py
    poetry run python visualizations/visualize_store_latency_scaling.py \\
        --input-csv benchmarks/results/store_latency_scaling.csv \\
        --output-dir visualizations/output/store_latency_scaling
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


STORE_STYLE: dict[str, dict] = {
    "naive":           {"color": "#E65100", "marker": "o", "linestyle": "-",  "label": "Naive (dict loop)"},
    "normalized":      {"color": "#FF8F00", "marker": "s", "linestyle": "--", "label": "Normalized"},
    "matrix":          {"color": "#1565C0", "marker": "^", "linestyle": "-",  "label": "Matrix"},
    "buffered-matrix": {"color": "#0288D1", "marker": "D", "linestyle": "--", "label": "Buffered Matrix"},
    "file":            {"color": "#2E7D32", "marker": "v", "linestyle": "-",  "label": "File-backed"},
    "mmap":            {"color": "#6A1B9A", "marker": "P", "linestyle": "--", "label": "MMap"},
}

FAST_STORES = {"matrix", "buffered-matrix", "file", "mmap"}


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.sort_values(["store", "checkpoint"])


def _savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _latency_plot(
    df: pd.DataFrame,
    out_dir: Path,
    filename: str,
    title: str,
    stores: set[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for store, style in STORE_STYLE.items():
        if stores is not None and store not in stores:
            continue
        rows = df[df["store"] == store].sort_values("checkpoint")
        if rows.empty:
            continue
        ax.plot(
            rows["checkpoint"],
            rows["p95_latency_ms"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=7,
            label=style["label"],
        )

    checkpoints = sorted(df["checkpoint"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(checkpoints)
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.set_xlabel("Number of records", fontsize=12)
    ax.set_ylabel("p95 search latency (ms)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    _savefig(fig, out_dir, filename)


def plot_all_stores(df: pd.DataFrame, out_dir: Path) -> None:
    _latency_plot(
        df, out_dir,
        filename="p95_latency_vs_records.png",
        title="Search Latency Scaling — All Stores (p95, log-log)",
    )


def plot_fast_stores(df: pd.DataFrame, out_dir: Path) -> None:
    _latency_plot(
        df, out_dir,
        filename="p95_latency_vs_records_fast_stores.png",
        title="Search Latency Scaling — Matrix-backed Stores (p95, log-log)",
        stores=FAST_STORES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize search latency scaling vs record count for all exact stores."
    )
    parser.add_argument(
        "--input-csv",
        default="benchmarks/results/store_latency_scaling.csv",
        help="Path to store_latency_scaling.csv (default: benchmarks/results/store_latency_scaling.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations/output/store_latency_scaling",
        help="Directory to write PNG files (default: visualizations/output/store_latency_scaling).",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_csv} ...")
    df = load_data(input_csv)
    print(f"  {len(df)} rows across stores: {sorted(df['store'].unique())}")
    print(f"  checkpoints: {sorted(df['checkpoint'].unique())}")

    print("\nGenerating plots ...")
    plot_all_stores(df, out_dir)
    plot_fast_stores(df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
