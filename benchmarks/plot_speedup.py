#!/usr/bin/env python3
"""
Plot speedup vs size curves for different cutoffs from one or more speedup_summary CSV files.

Usage:
    python plot_speedup.py [file1.csv [file2.csv ...]]
    (default: speedup_summary.csv if no arguments)

When multiple files are given, each file's curves get a distinct legend suffix:
  - From filename: speedup_summary_04.csv -> "04"; my_run_abc.csv -> "abc"
  - Fallback: "1", "2", "3", ... (by order of files)

Output: speedup_plot.png
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

CUTOFFS = [0.1, 0.2, 0.35]
# One color family per file (same file = similar colors); cutoff low→high = light→dark
CMAP_FAMILIES = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Purples]
CUTOFF_SHADES = (0.35, 0.55, 0.85)  # light, mid, dark


def suffix_from_filename(path: str, index: int) -> str:
    """Derive legend suffix from filename: part after last underscore before .csv, or '1','2',..."""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    if "_" in stem:
        suffix = stem.split("_")[-1]
        if suffix:
            return suffix
    return str(index + 1)


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = ["speedup_summary.csv"]

    for p in paths:
        if not os.path.isfile(p):
            print(f"Error: not a file: {p}", file=sys.stderr)
            sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for file_index, path in enumerate(paths):
        suffix = suffix_from_filename(path, file_index)
        df = pd.read_csv(path)
        df = df[df["cutoff"].isin(CUTOFFS)]
        cmap = CMAP_FAMILIES[file_index % len(CMAP_FAMILIES)]

        for i, cutoff in enumerate(CUTOFFS):
            subset = df[df["cutoff"] == cutoff].sort_values("size")
            if len(subset) == 0:
                continue
            color = cmap(CUTOFF_SHADES[min(i, len(CUTOFF_SHADES) - 1)])
            label = f"cutoff={cutoff} ({suffix})"
            ax.plot(
                subset["size"],
                subset["speedup"],
                marker="o",
                label=label,
                color=color,
                linewidth=2,
            )

    ax.set_xlabel("Dataset Size (molecules)", fontsize=12)
    ax.set_ylabel("Speedup (RDKit / nvMolKit)", fontsize=12)
    ax.set_title("nvMolKit Butina Clustering Speedup vs RDKit", fontsize=14)
    ax.legend(title="Cutoff (file)", loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("speedup_plot.png", dpi=150)
    print("Saved speedup_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
