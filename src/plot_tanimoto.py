import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['figure.figsize'] = (10, 8)

color_nv_green = "#76b900"
color_rdkit_blue = "#4300B9"
color_a100_magenta = "#EE3377"

# ----------------------------------------------------------
# size scans
# ---------------------------------------------------------
num_fps = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]

rdkit_h100_seconds = [0.310, 1.27, 2.98, 5.45, 8.51, 12.3, 16.8, 21.0]
rdkit_h100_error = [0.015, 0.03, 0.02, 0.12, 0.30, 0.1, 0.1, 0.1]
rdcu_h100_seconds = [0.00113, 0.00149, 0.00204, 0.00279, 0.00373, 0.00496, 0.00622, 0.00787]
rdcu_h100_error = [0.00001, 0.0, 0.00001, 0.00001, 0.00001, 0.00002, 0.00005, 0.00005]

rdcu_h100_tensor = [0.00120, 0.00139, 0.00173, 0.00226, 0.00273, 0.00351, 0.00447, 0.00563]
rdcu_h100_tensor_error = [0.00007, 0.00005, 0.00003, 0.00007, 0.00006, 0.00011, 0.00022, 0.00046]

rdkit_a100_seconds = [0.431, 1.76, 3.95, 7.06, 11.1, 15.9, 21.7, 28.6]
rdkit_a100_error = [0.016, 0.03, 0.02, 0.06, 0.2, 0.1, 0.0, 0.2]
rdcu_a100_seconds = [0.00117, 0.00148, 0.00217, 0.00295, 0.00410, 0.00563, 0.00724, 0.012]
rdcu_a100_error = [0.00001, 0.00003, 0.00002, 0.00022, 0.00003, 0.00004, 0.00005, 0.004]


# Compute total comparisons (n^2) for each fingerprint count
num_fps_array = np.array(num_fps, dtype=float)
num_comparisons = num_fps_array ** 2

# Convert seconds and errors to numpy arrays
rdkit_h100_seconds_arr = np.array(rdkit_h100_seconds, dtype=float)
rdkit_h100_error_arr = np.array(rdkit_h100_error, dtype=float)
rdcu_h100_seconds_arr = np.array(rdcu_h100_seconds, dtype=float)
rdcu_h100_error_arr = np.array(rdcu_h100_error, dtype=float)
rdcu_h100_tensor_arr = np.array(rdcu_h100_tensor, dtype=float)
rdcu_h100_tensor_error_arr = np.array(rdcu_h100_tensor_error, dtype=float)

rdkit_a100_seconds_arr = np.array(rdkit_a100_seconds, dtype=float)
rdkit_a100_error_arr = np.array(rdkit_a100_error, dtype=float)
rdcu_a100_seconds_arr = np.array(rdcu_a100_seconds, dtype=float)
rdcu_a100_error_arr = np.array(rdcu_a100_error, dtype=float)

# Throughput (comparisons per second)
rdkit_h100_throughput = num_comparisons / rdkit_h100_seconds_arr
rdcu_h100_throughput = num_comparisons / rdcu_h100_seconds_arr
rdcu_h100_tensor_throughput = num_comparisons / rdcu_h100_tensor_arr
rdkit_a100_throughput = num_comparisons / rdkit_a100_seconds_arr
rdcu_a100_throughput = num_comparisons / rdcu_a100_seconds_arr

# Error propagation: y = N / t -> sigma_y = N * sigma_t / t^2
rdkit_h100_throughput_std = num_comparisons * rdkit_h100_error_arr / (rdkit_h100_seconds_arr ** 2)
rdcu_h100_throughput_std = num_comparisons * rdcu_h100_error_arr / (rdcu_h100_seconds_arr ** 2)
rdcu_h100_tensor_throughput_std = num_comparisons * rdcu_h100_tensor_error_arr / (rdcu_h100_tensor_arr ** 2)
rdkit_a100_throughput_std = num_comparisons * rdkit_a100_error_arr / (rdkit_a100_seconds_arr ** 2)
rdcu_a100_throughput_std = num_comparisons * rdcu_a100_error_arr / (rdcu_a100_seconds_arr ** 2)

# ----------------------------------------------------------
# H100 plots
# ---------------------------------------------------------
plt.errorbar(num_fps, rdcu_h100_throughput, yerr=rdcu_h100_throughput_std, label="nvmolkit", fmt="o-", capsize=5)
plt.errorbar(num_fps, rdkit_h100_throughput, yerr=rdkit_h100_throughput_std, label="rdkit", fmt="o-", capsize=5)
plt.errorbar(num_fps, rdcu_h100_tensor_throughput, yerr=rdcu_h100_tensor_throughput_std, label="nvmolkit (tensor)", fmt="o-", capsize=5)
plt.xlabel("Number of fingerprints")
plt.ylabel("Comparisons per second")
plt.title("Throughput - DGX H100 box")
plt.xticks(num_fps, [str(n) for n in num_fps])
plt.legend()
plt.show()

# Semilogy for H100
plt.semilogy(num_fps, rdcu_h100_throughput, label="nvmolkit", marker="o", linestyle="-")
plt.semilogy(num_fps, rdkit_h100_throughput, label="rdkit", marker="o", linestyle="-")
plt.semilogy(num_fps, rdcu_h100_tensor_throughput, label="nvmolkit (tensor)", marker="o", linestyle="-")
plt.xlabel("Number of fingerprints")
plt.ylabel("Comparisons per second")
plt.title("Semilogy - DGX H100 box")
plt.xticks(num_fps, [str(n) for n in num_fps])
plt.legend()
plt.show()

# Speedup for H100 (nvmolkit over rdkit)
h100_speedup = rdcu_h100_throughput / rdkit_h100_throughput
h100_tensor_speedup = rdcu_h100_tensor_throughput / rdkit_h100_throughput
# Propagate uncertainty for ratio y = a/b: sigma_y = y * sqrt((sa/a)^2 + (sb/b)^2)
h100_speedup_std = h100_speedup * np.sqrt(
    (rdcu_h100_throughput_std / rdcu_h100_throughput) ** 2 +
    (rdkit_h100_throughput_std / rdkit_h100_throughput) ** 2
)
h100_tensor_speedup_std = h100_tensor_speedup * np.sqrt(
    (rdcu_h100_tensor_throughput_std / rdcu_h100_tensor_throughput) ** 2 +
    (rdkit_h100_throughput_std / rdkit_h100_throughput) ** 2
)

plt.errorbar(num_fps, h100_speedup, yerr=h100_speedup_std, label="speedup", fmt="o-", capsize=5)
plt.errorbar(num_fps, h100_tensor_speedup, yerr=h100_tensor_speedup_std, label="speedup (tensor)", fmt="o-", capsize=5)

plt.xlabel("Number of fingerprints")
plt.ylabel("Speedup ratio over rdkit")
plt.title("Speedup - DGX H100 box")
plt.ylim(0, h100_tensor_speedup.max() * 1.1)
plt.xticks(num_fps, [str(n) for n in num_fps])
plt.show()

# ----------------------------------------------------------
# A100 plots
# ---------------------------------------------------------
plt.errorbar(num_fps, rdcu_a100_throughput, yerr=rdcu_a100_throughput_std, label="nvmolkit", fmt="o-", capsize=5)
plt.errorbar(num_fps, rdkit_a100_throughput, yerr=rdkit_a100_throughput_std, label="rdkit", fmt="o-", capsize=5)
plt.xlabel("Number of fingerprints")
plt.ylabel("Comparisons per second")
plt.title("Throughput - DGX A100 box")
plt.xticks(num_fps, [str(n) for n in num_fps])
plt.legend()
plt.show()

# Semilogy for A100
plt.semilogy(num_fps, rdcu_a100_throughput, label="nvmolkit", marker="o", linestyle="-")
plt.semilogy(num_fps, rdkit_a100_throughput, label="rdkit", marker="o", linestyle="-")
plt.xlabel("Number of fingerprints")
plt.ylabel("Comparisons per second")
plt.title("Semilogy - DGX A100 box")
plt.xticks(num_fps, [str(n) for n in num_fps])
plt.legend()
plt.show()

# Speedup for A100 (nvmolkit over rdkit)
a100_speedup = rdcu_a100_throughput / rdkit_a100_throughput
# Propagate uncertainty for ratio
a100_speedup_std = a100_speedup * np.sqrt(
    (rdcu_a100_throughput_std / rdcu_a100_throughput) ** 2 +
    (rdkit_a100_throughput_std / rdkit_a100_throughput) ** 2
)

plt.errorbar(num_fps, a100_speedup, yerr=a100_speedup_std, label="speedup", fmt="o-", capsize=5)
plt.xlabel("Number of fingerprints")
plt.ylabel("Speedup ratio over rdkit")
plt.title("Speedup - DGX A100 box")
plt.ylim(0, a100_speedup.max() * 1.1)
plt.xticks(num_fps, [str(n) for n in num_fps])
plt.show()



plt.figure(figsize=(10, 6))
plt.errorbar(num_fps, h100_speedup, yerr=h100_speedup_std, label="H100 DGX speedup", fmt="o-", capsize=5)
plt.errorbar(num_fps, a100_speedup, yerr=a100_speedup_std, label="A100 DGX speedup", fmt="s-", capsize=5)
plt.xlabel("Number of fingerprints")
plt.ylabel("Speedup ratio over rdkit")
plt.title("Speedup Comparison: H100 vs A100 (nvmolkit over rdkit)")
plt.ylim(0, max(h100_speedup.max(), a100_speedup.max()) * 1.1)
plt.xticks(num_fps, [str(n) for n in num_fps])
plt.legend()
plt.show()

# Make plot with a100 and h100 tensor
plt.errorbar(num_fps, h100_tensor_speedup, yerr=h100_tensor_speedup_std,  fmt="o-", capsize=5, color=color_nv_green)
#plt.errorbar(num_fps, a100_speedup, yerr=a100_speedup_std, label="A100 DGX speedup", fmt="s-", capsize=5)
plt.xlabel("Number of fingerprints")
plt.ylabel("Speedup ratio over rdkit")
plt.title("Speedup Comparison: H100 vs RDKit")
plt.ylim(0, max(h100_tensor_speedup.max(), a100_speedup.max()) * 1.1)
plt.xticks(num_fps, [str(n) for n in num_fps])
plt.legend()
plt.show()
# ----------------------------------------------------------
# comparison across programs
# ----------------------------------------------------------

n = 16000


xticks = ["RDKit", "PyTorch\n(extrap. n=1000)", "FPSim-Cupy on H100", "scikit-fingerprints", "Chemfp", "nvmolkit A100", "nvmolkit H100"]
seconds = [21.0, 7.472025, 6.95, 6.82, 2.17, 0.00724, 0.00563]
errors = [0.1, 0.021521, 0.03615155, 0.245, 0.01037135, 0.00005, 0.00046]

throughput = [ n * n / sec for sec in seconds ]
# Use percentage errors for throughput error bars
throughput_error = [ t * (err / sec) for t, err, sec in zip(throughput, errors, seconds) ]

import matplotlib.ticker as mticker
plt.figure(figsize=(14, 6))
bar_colors = [color_rdkit_blue, color_rdkit_blue, color_rdkit_blue, color_rdkit_blue, color_rdkit_blue, color_a100_magenta, color_nv_green]  # Added orange for PyTorch
bars = plt.bar(xticks, throughput, yerr=throughput_error, capsize=8, color=bar_colors)
plt.ylabel("Throughput (pairs/sec)", fontsize=16)
plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(10_000_000_000))
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1e9)}B' if x >= 1e9 else f'{int(x):,}'))

# Annotate the first 5 data points with the value
for i in range(5):
    bar = bars[i]
    height = bar.get_height()
    plt.annotate(
        f"{int(height/1e6):,}M",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha='center', va='bottom', fontsize=18, fontweight='bold'
    )

plt.title("Tanimoto All-vs-All (n=16,000, 1024 bits) - Throughput Comparison", fontsize=18)
plt.ylim(0, max(throughput) * 1.1)
plt.xticks(fontsize=10)  # Smaller font size for x-axis labels
plt.show()

# Isolate the first 5 data points. Make bar plots for CPU-based implementations
plt.figure(figsize=(10, 6))  # Add figure size control for second plot
first_5_xticks = xticks[:5]
first_5_throughput = throughput[:5]
first_5_throughput_error = throughput_error[:5]
first_5_colors = bar_colors[:5]
first_5_bars = plt.bar(first_5_xticks, first_5_throughput, yerr=first_5_throughput_error, capsize=8, color=first_5_colors)
plt.ylabel("Throughput (pairs/sec)", fontsize=16)
plt.title("Tanimoto All-vs-All (n=16,000, 1024 bits) - CPU/GPU Implementations", fontsize=18)
plt.ylim(0, max(first_5_throughput) * 1.1)
plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(10_000_000))
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1e6)}M' if x >= 1e6 else f'{int(x):,}'))

# Annotate all 5 bars
for i in range(5):
    bar = first_5_bars[i]
    height = bar.get_height()
    plt.annotate(
        f"{int(height/1e6):,}M",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha='center', va='bottom', fontsize=18, fontweight='bold'
    )
plt.xticks(fontsize=10)  # Smaller font size for x-axis labels
plt.show()

# Speedup plot vs rdkit
# Compute speedup and its error relative to RDKit
rdkit_time = seconds[0]
rdkit_error = errors[0]
speedup = [rdkit_time / t for t in seconds]
# Propagate error: (ΔS/S)^2 = (ΔT/T)^2 + (Δt/t)^2
import numpy as np
speedup_error = [
    s * np.sqrt( (rdkit_error/rdkit_time)**2 + (err/t)**2 )
    for s, t, err in zip(speedup, seconds, errors)
]

plt.figure()
plt.semilogy(xticks, speedup, marker="o", label="speedup")
plt.errorbar(xticks, speedup, yerr=speedup_error, fmt="none", capsize=5, color="C0")
plt.ylabel("Speedup ratio over rdkit (log scale)")
plt.title("Tanimoto All-vs-All (n=16,000, 1024 bits) - Speedup Comparison")
plt.ylim(1, max(speedup) * 1.1)
plt.show()
