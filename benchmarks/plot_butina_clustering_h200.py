import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


mpl.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 24,
    'lines.markersize': 8,
    'lines.linewidth': 2,
    'figure.figsize': (14, 8)
})

# v3 = reds (light to dark for cutoff 0.1, 0.2, 0.35); v4 = blues (light to dark)
COLORS_V3 = [plt.cm.Reds(0.35), plt.cm.Reds(0.55), plt.cm.Reds(0.85)]
COLORS_V4 = [plt.cm.Blues(0.35), plt.cm.Blues(0.55), plt.cm.Blues(0.85)]
LEGEND_KW = {"bbox_to_anchor": (1.02, 1), "loc": "upper left", "fontsize": 14}


z_95_conf = 1.96
n_trials = 3
CUTOFFS = [0.1, 0.2, 0.35]

# Load v3 (H200 xeon4314); drop size 60000 to match v4 result sizes
df_v3 = pd.read_csv("butina_results_h200_xeon4314_v03_final.csv")
df_v3 = df_v3[df_v3['size'] != 60000].copy()
df_v3_rdkit = df_v3[df_v3['max_neighborlist_size'] == 64].copy()
df_v3_nvmol_best = df_v3.loc[df_v3.groupby(['size', 'cutoff'])['nvmol_time_ms'].idxmin()].copy()

# Load v4 (results.csv): keep only NL 64 and 128 to match v3
df_v4_raw = pd.read_csv("results_v4.csv")
df_v4 = df_v4_raw[df_v4_raw['max_neighborlist_size'].isin([64, 128])].copy()
df_v4_rdkit = df_v4[df_v4['max_neighborlist_size'] == 64].copy()
df_v4_nvmol_best = df_v4.loc[df_v4.groupby(['size', 'cutoff'])['nvmol_time_ms'].idxmin()].copy()

# List of (df_rdkit, df_nvmol_best, df_full, suffix) for each dataset
datasets = [
    (df_v3_rdkit, df_v3_nvmol_best, df_v3, "v3"),
    (df_v4_rdkit, df_v4_nvmol_best, df_v4, "v4"),
]

# Speedup plot: both v3 and v4
plt.figure()
for ds_idx, (df_rdkit, df_nvmol_best, _df_full, suffix) in enumerate(datasets):
    pal = COLORS_V3 if suffix == "v3" else COLORS_V4
    for i, cutoff in enumerate(CUTOFFS):
        rdkit_subset = df_rdkit[df_rdkit['cutoff'] == cutoff].set_index('size')
        nvmol_subset = df_nvmol_best[df_nvmol_best['cutoff'] == cutoff].set_index('size')
        common_sizes = rdkit_subset.index.intersection(nvmol_subset.index)
        if len(common_sizes) == 0:
            continue
        rdkit_time = rdkit_subset.loc[common_sizes, 'rdkit_time_ms']
        nvmol_time = nvmol_subset.loc[common_sizes, 'nvmol_time_ms']
        rdkit_std = rdkit_subset.loc[common_sizes, 'rdkit_std_ms']
        nvmol_std = nvmol_subset.loc[common_sizes, 'nvmol_std_ms']
        speedup = rdkit_time / nvmol_time
        rdkit_ci = rdkit_std * z_95_conf / math.sqrt(n_trials)
        nvmol_ci = nvmol_std * z_95_conf / math.sqrt(n_trials)
        speedup_err = speedup * ((rdkit_ci / rdkit_time)**2 + (nvmol_ci / nvmol_time)**2)**0.5
        plt.errorbar(
            common_sizes, speedup, yerr=speedup_err, fmt='o-',
            label=f'Cutoff {cutoff} ({suffix})', color=pal[i], capsize=4
        )
plt.xlabel('Number of Molecules')
plt.ylabel('Speedup (nvMolKit / RDKit)')
plt.title('Butina Clustering Speedup on an H200 over CPU RDKit')
plt.legend(**LEGEND_KW)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('butina_speedup_h200.png', dpi=150, bbox_inches='tight')
plt.show()

# Raw time plot: RDKit (CPU) times — v3 and v4
plt.figure()
for df_rdkit, _df_nvmol_best, _df_full, suffix in datasets:
    pal = COLORS_V3 if suffix == "v3" else COLORS_V4
    for i, cutoff in enumerate(CUTOFFS):
        subset = df_rdkit[df_rdkit['cutoff'] == cutoff]
        if len(subset) == 0:
            continue
        rdkit_ci = subset['rdkit_std_ms'] * z_95_conf / math.sqrt(n_trials)
        plt.errorbar(
            subset['size'], subset['rdkit_time_ms'], yerr=rdkit_ci,
            label=f'RDKit CPU Cutoff {cutoff} ({suffix})', color=pal[i], capsize=4
        )
plt.xlabel('Number of Molecules')
plt.ylabel('Time (ms)')
plt.title('Butina Clustering Time (RDKit CPU)')
plt.legend(**LEGEND_KW)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('butina_time_rdkit_h200.png', dpi=150, bbox_inches='tight')
plt.show()

# Raw time plot: nvMolKit (GPU) times — v3 and v4, NL 64 and 128
plt.figure()
# Linestyles: v3 NL64/128 = solid/dashed; v4 NL64/128 = dashdot/dotted
for df_rdkit, df_nvmol_best, df_full, suffix in datasets:
    pal = COLORS_V3 if suffix == "v3" else COLORS_V4
    styles = [(64, "-"), (128, "--")] if suffix == "v3" else [(64, "-."), (128, ":")]
    for i, cutoff in enumerate(CUTOFFS):
        for nl_size, linestyle in styles:
            subset = df_full[(df_full['cutoff'] == cutoff) & (df_full['max_neighborlist_size'] == nl_size)]
            if len(subset) == 0:
                continue
            nvmol_ci = subset['nvmol_std_ms'] * z_95_conf / math.sqrt(n_trials)
            label = f'Cutoff {cutoff}, NL {nl_size} ({suffix})'
            plt.errorbar(
                subset['size'], subset['nvmol_time_ms'], yerr=nvmol_ci,
                color=pal[i], linestyle=linestyle, marker='o',
                label=label, capsize=4
            )
plt.xlabel('Number of Molecules')
plt.ylabel('Time (ms)')
plt.title('Butina Clustering Time (nvMolKit GPU)')
plt.legend(**LEGEND_KW)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig('butina_time_nvmol_h200.png', dpi=150, bbox_inches='tight')
plt.show()
