#!/usr/bin/env python3
"""
Plot nvMolKit GPU time vs dataset size: v3 vs v4 results.
Uses only neighbor list sizes 64 and 128; for each (size, cutoff) uses best time
(min nvmol_time_ms) over those. Ignores RDKit. v3 = reds, v4 = blues.
Prints to terminal: percentage improvement v4 vs v3 (positive = v4 faster).
"""

import pandas as pd
import matplotlib.pyplot as plt

# Hardcoded input files (run from directory that contains them, or set full paths)
RESULTS_V3 = "results_v3.csv"
RESULTS_V4 = "results_v4.csv"
OUTPUT_PNG = "nvmol_time_v3_v4.png"

CUTOFFS = [0.1, 0.2, 0.35]
NEIGHBORLIST_SIZES = [64, 128]  # only keep these when selecting best nvMol time
COLORS_V3 = [plt.cm.Reds(0.35), plt.cm.Reds(0.55), plt.cm.Reds(0.85)]
COLORS_V4 = [plt.cm.Blues(0.35), plt.cm.Blues(0.55), plt.cm.Blues(0.85)]
LEGEND_KW = {"bbox_to_anchor": (1.02, 1), "loc": "upper left", "fontsize": 14}


def best_nvmol_per_size_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    """For each (size, cutoff), keep row with min nvmol_time_ms."""
    idx = df.groupby(["size", "cutoff"])["nvmol_time_ms"].idxmin()
    return df.loc[idx, ["size", "cutoff", "nvmol_time_ms"]].sort_values(
        ["cutoff", "size"]
    )


def main():
    df_v3 = pd.read_csv(RESULTS_V3)
    df_v4 = pd.read_csv(RESULTS_V4)
    df_v3 = df_v3[df_v3["max_neighborlist_size"].isin(NEIGHBORLIST_SIZES)]
    df_v4 = df_v4[df_v4["max_neighborlist_size"].isin(NEIGHBORLIST_SIZES)]
    best_v3 = best_nvmol_per_size_cutoff(df_v3)
    best_v4 = best_nvmol_per_size_cutoff(df_v4)

    # Plot: nvMolKit time vs size, v3 reds / v4 blues
    fig, ax = plt.subplots(figsize=(14, 8))
    for (best, colors, label_prefix) in [
        (best_v3, COLORS_V3, "v3"),
        (best_v4, COLORS_V4, "v4"),
    ]:
        for i, cutoff in enumerate(CUTOFFS):
            sub = best[best["cutoff"] == cutoff].sort_values("size")
            if len(sub) == 0:
                continue
            ax.plot(
                sub["size"],
                sub["nvmol_time_ms"],
                marker="o",
                label=f"cutoff={cutoff} ({label_prefix})",
                color=colors[i],
                linewidth=2,
            )

    ax.set_xlabel("Dataset Size (molecules)", fontsize=12)
    ax.set_ylabel("nvMolKit Time (ms)", fontsize=12)
    ax.set_title("nvMolKit GPU Time vs Dataset Size (v3 vs v4)", fontsize=14)
    ax.legend(**LEGEND_KW)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUTPUT_PNG}")

    # Percentage improvement v4 vs v3 (positive = v4 faster)
    merged = best_v3.merge(
        best_v4,
        on=["size", "cutoff"],
        suffixes=("_v3", "_v4"),
        how="inner",
    )
    if merged.empty:
        print("No common (size, cutoff) points between v3 and v4.")
        return

    merged["improvement_pct"] = (
        (merged["nvmol_time_ms_v3"] - merged["nvmol_time_ms_v4"])
        / merged["nvmol_time_ms_v3"]
        * 100
    )

    print("\n--- Improvement v4 vs v3 (positive = v4 faster) ---")
    for cutoff in CUTOFFS:
        m = merged[merged["cutoff"] == cutoff]
        if m.empty:
            continue
        pct = m["improvement_pct"].mean()
        print(f"  Cutoff {cutoff}: mean improvement = {pct:+.1f}%")
    overall = merged["improvement_pct"].mean()
    print(f"  Overall (all sizes, cutoffs): mean improvement = {overall:+.1f}%")
    print("\nPer (size, cutoff):")
    for cutoff in CUTOFFS:
        m = merged[merged["cutoff"] == cutoff].sort_values("size")
        if m.empty:
            continue
        print(f"  cutoff={cutoff}:")
        for _, row in m.iterrows():
            print(f"    size={row['size']:6.0f}  improvement = {row['improvement_pct']:+.1f}%")


if __name__ == "__main__":
    main()
