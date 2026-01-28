#!/usr/bin/env python3
"""Plot speedup vs size curves for different cutoffs from speedup_summary.csv"""

import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("speedup_summary.csv")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot a curve for each cutoff (only realistic cutoffs)
    cutoffs = [0.1, 0.2, 0.35]
    df = df[df["cutoff"].isin(cutoffs)]
    n = len(cutoffs)
    colors = plt.cm.viridis([i / max(n - 1, 1) for i in range(n)])
    
    for cutoff, color in zip(cutoffs, colors):
        subset = df[df["cutoff"] == cutoff].sort_values("size")
        ax.plot(subset["size"], subset["speedup"], marker="o", label=f"cutoff={cutoff}", color=color, linewidth=2)
    
    ax.set_xlabel("Dataset Size (molecules)", fontsize=12)
    ax.set_ylabel("Speedup (RDKit / nvMolKit)", fontsize=12)
    ax.set_title("nvMolKit Butina Clustering Speedup vs RDKit", fontsize=14)
    ax.legend(title="Distance Cutoff", loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("speedup_plot.png", dpi=150)
    print("Saved speedup_plot.png")
    plt.show()

if __name__ == "__main__":
    main()

