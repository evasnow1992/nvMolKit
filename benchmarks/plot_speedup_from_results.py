#!/usr/bin/env python3
"""
Plot speedup vs size from one or more results.csv files, with error bars.

For each (size, cutoff), uses the row with best speedup (max over max_neighborlist_size)
and computes speedup std from rdkit_std_ms and nvmol_std_ms via error propagation.

Usage:
    python plot_speedup_from_results.py [file1.csv [file2.csv ...]]
    (default: results.csv if no arguments)

Legend suffix from filename: results_03.csv -> "03"; fallback "1","2",...

Output: speedup_plot_with_err.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CUTOFFS = [0.1, 0.2, 0.35]
TITLE = "nvMolKit Butina Clustering Speedup vs RDKit"


def suffix_from_filename(path: str, index: int) -> str:
    """Derive legend suffix from filename: part after last underscore before .csv."""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    if "_" in stem:
        suffix = stem.split("_")[-1]
        if suffix:
            return suffix
    return str(index + 1)


def speedup_std(rdkit_time: float, rdkit_std: float, nvmol_time: float, nvmol_std: float) -> float:
    """Propagation of uncertainty for speedup = rdkit_time / nvmol_time."""
    if nvmol_time <= 0:
        return 0.0
    r = rdkit_time / nvmol_time
    # sigma_R^2 = (sigma_A/B)^2 + (A*sigma_B/B^2)^2
    var = (rdkit_std / nvmol_time) ** 2 + (rdkit_time * nvmol_std / (nvmol_time ** 2)) ** 2
    return np.sqrt(var)


def best_speedup_per_size_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    """For each (size, cutoff), keep row with max speedup; add speedup_std column."""
    df = df.copy()
    if "speedup" not in df.columns:
        df["speedup"] = df["rdkit_time_ms"] / df["nvmol_time_ms"]
    df["speedup_std"] = df.apply(
        lambda row: speedup_std(
            row["rdkit_time_ms"],
            row["rdkit_std_ms"],
            row["nvmol_time_ms"],
            row["nvmol_std_ms"],
        ),
        axis=1,
    )
    df = df[df["cutoff"].isin(CUTOFFS)]
    idx = df.groupby(["size", "cutoff"])["speedup"].idxmax()
    best = df.loc[idx, ["size", "cutoff", "speedup", "speedup_std"]].sort_values(
        ["cutoff", "size"]
    )
    return best


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = ["results.csv"]

    for p in paths:
        if not os.path.isfile(p):
            print(f"Error: not a file: {p}", file=sys.stderr)
            sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 6))

    n_files = len(paths)
    n_curves = n_files * len(CUTOFFS)
    colors = plt.cm.viridis([i / max(n_curves - 1, 1) for i in range(n_curves)])
    idx = 0

    for file_index, path in enumerate(paths):
        suffix = suffix_from_filename(path, file_index)
        df = pd.read_csv(path)
        best = best_speedup_per_size_cutoff(df)

        for cutoff in CUTOFFS:
            sub = best[best["cutoff"] == cutoff].sort_values("size")
            if len(sub) == 0:
                continue
            label = f"cutoff={cutoff} ({suffix})"
            ax.errorbar(
                sub["size"],
                sub["speedup"],
                yerr=sub["speedup_std"],
                marker="o",
                label=label,
                color=colors[idx],
                linewidth=2,
                capsize=4,
            )
            idx += 1

    ax.set_xlabel("Dataset Size (molecules)", fontsize=12)
    ax.set_ylabel("Speedup (RDKit / nvMolKit)", fontsize=12)
    ax.set_title(TITLE, fontsize=14)
    ax.legend(title="Cutoff (file)", loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "speedup_plot_with_err.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
