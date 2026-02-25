# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.ML.Cluster.Butina import ClusterData
import torch

from nvmolkit.clustering import butina as butina_nvmol
from nvmolkit.fingerprints import MorganFingerprintGenerator as nvmolMorganGen
from nvmolkit.similarity import crossTanimotoSimilarity


def check_butina_correctness(hit_mat, clusts):
    hit_mat = hit_mat.clone()
    seen = set()

    for i, clust in enumerate(clusts):
        assert len(clust) > 0, "Empty cluster found"
        clust_size = len(clust)

        if clust_size == 1:
            remaining_items = []
            for remaining_clust in clusts[i:]:
                assert len(remaining_clust) == 1, "Expected all remaining clusters to be singletons"
                remaining_items.append(remaining_clust[0])

            remaining_set = set(remaining_items)
            assert len(remaining_set) == len(remaining_items), "Duplicate items in singleton clusters"
            assert remaining_set.isdisjoint(seen), "Singleton item was already seen"
            seen.update(remaining_set)
            break
        counts = hit_mat.sum(-1)
        assert clust_size == counts.max(), f"Cluster size {clust_size} doesn't match max available count {counts.max()}"
        for item in clust:
            assert item not in seen, f"Point {item} assigned to multiple clusters"
            seen.add(item)
            hit_mat[item, :] = False
            hit_mat[:, item] = False
    assert len(seen) == hit_mat.shape[0]


def get_distance_matrix(molecules):
    nvmol_gen = nvmolMorganGen(radius=2, fpSize=1024)
    nvmol_fps = nvmol_gen.GetFingerprints(molecules, 10)
    sim_matrix = crossTanimotoSimilarity(nvmol_fps).torch()
    return 1.0 - sim_matrix


def resize_and_fill(distance_mat: torch.Tensor, want_size):
    current_size = distance_mat.shape[0]
    if current_size >= want_size:
        return distance_mat[:want_size, :want_size].contiguous()
    full_mat = torch.rand(want_size, want_size, dtype=distance_mat.dtype, device=distance_mat.device)
    full_mat = torch.abs(full_mat - full_mat.T).clip(0.01, 0.99)
    full_mat.fill_diagonal_(0.0)
    full_mat[:current_size, :current_size] = distance_mat
    return full_mat


def time_it(func, runs=3, warmups=1):
    import time

    for _ in range(warmups):
        func()
    times = []
    for _ in range(runs):
        start = time.time_ns()
        func()
        end = time.time_ns()
        times.append(end - start)
    avg_time = sum(times) / runs
    std_time = (sum((t - avg_time) ** 2 for t in times) / runs) ** 0.5
    return avg_time / 1.0e6, std_time / 1.0e6  # return in milliseconds


def bench_rdkit(data, threshold):
    return time_it(lambda: ClusterData(data, len(data), threshold, isDistData=True, reordering=True))

def bench_nvmol_inner(data, threshold, neighborlist_max_size):
    butina_nvmol(data, threshold, neighborlist_max_size=neighborlist_max_size)
    torch.cuda.synchronize()

DEFAULT_INPUT_FILE = "benchmarks/data/chembl_10k.smi"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python butina_clustering_bench.py <do_rdkit (0 or 1)> [input_smiles_file]")
        print(f"  Default input file: {DEFAULT_INPUT_FILE}")
        sys.exit(1)
    do_rdkit = sys.argv[1] != "0"
    input_data = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_INPUT_FILE

    print(f"Loading molecules from {input_data}")
    with open(input_data, "r") as f:
        smis = [line.strip() for line in f.readlines()]
    mols = [MolFromSmiles(smi, sanitize=True) for smi in smis]
    mols = [mol for mol in mols if mol is not None]
    max_size = len(mols)
    print(f"Loaded {max_size} valid molecules")

    dists = get_distance_matrix(mols)

    # Only use sizes that fit within the available data (no padding)
    all_sizes = [5000, 10000, 20000, 30000, 40000, 50000, 60000]
    sizes = [s for s in all_sizes if s <= max_size]
    print(f"Will benchmark sizes: {sizes}")

    cutoffs = [0.1, 0.2, 0.35]
    max_nl_sizes = [8, 16, 32, 64, 128]
    results = []

    try:
        for size in sizes:
            for cutoff in cutoffs:
                    for max_nl in max_nl_sizes:
                        print(f"Running size {size} cutoff {cutoff} max_nl {max_nl}")
                        dist_mat = resize_and_fill(dists, size)
                        if do_rdkit:
                            dist_mat_numpy = dist_mat.cpu().numpy()
                            rdkit_time, rdk_std = bench_rdkit(dist_mat_numpy, cutoff)
                        else:
                            rdkit_time = 0.0
                            rdk_std = 0.0
                        nvmol_time, nvmol_std = time_it(lambda: bench_nvmol_inner(dist_mat, cutoff, max_nl))

                        # Verify correctness
                        nvmol_res = butina_nvmol(dist_mat, cutoff, neighborlist_max_size=max_nl).torch()
                        torch.cuda.synchronize()
                        nvmol_clusts = [tuple(torch.argwhere(nvmol_res == i).flatten().tolist()) for i in range(nvmol_res.max() + 1)]
                        check_butina_correctness(dist_mat <= cutoff, nvmol_clusts)

                        results.append(
                            {
                                "size": size,
                                "cutoff": cutoff,
                                "max_neighborlist_size": max_nl,
                                "rdkit_time_ms": rdkit_time,
                                "rdkit_std_ms": rdk_std,
                                "nvmol_time_ms": nvmol_time,
                                "nvmol_std_ms": nvmol_std,
                            }
                        )
    except Exception as e:
        print(f"Got exception: {e}, exiting early")
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("results.csv", index=False)

    # Generate speedup summary table (best speedup per size/cutoff combination)
    if do_rdkit and len(results) > 0:
        df["speedup"] = df["rdkit_time_ms"] / df["nvmol_time_ms"]

        # For each (size, cutoff), find the row with maximum speedup
        idx = df.groupby(["size", "cutoff"])["speedup"].idxmax()
        best_df = df.loc[idx, ["size", "cutoff", "speedup", "max_neighborlist_size"]].copy()
        best_df = best_df.rename(columns={"max_neighborlist_size": "best_neighborlist_size"})
        best_df = best_df.sort_values(["size", "cutoff"]).reset_index(drop=True)

        print("\n" + "=" * 60)
        print("SPEEDUP SUMMARY (max speedup per size/cutoff)")
        print("=" * 60)
        print(best_df.to_string(index=False))
        best_df.to_csv("speedup_summary.csv", index=False)
        print("\nSpeedup summary saved to speedup_summary.csv")
