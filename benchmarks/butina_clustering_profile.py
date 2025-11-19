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

"""
Profiling script for Butina clustering with NVTX annotations.

This script is designed to be used with NVIDIA Nsight Systems for detailed
performance analysis. Run with:
    nsys profile -o butina_profile python butina_clustering_profile.py <size> <cutoff> [warmup]

Usage:
    python butina_clustering_profile.py <size> <cutoff> [warmup]

Arguments:
    size: Number of molecules to cluster
    cutoff: Distance cutoff for Butina clustering
    warmup: Optional, 1 to enable warmup run, 0 to disable (default: 0)

Example:
    python butina_clustering_profile.py 10000 0.2 1
"""

import sys

import nvtx
import torch
from rdkit.Chem import MolFromSmiles

from nvmolkit.clustering import butina as butina_nvmol
from nvmolkit.fingerprints import MorganFingerprintGenerator as nvmolMorganGen
from nvmolkit.similarity import crossTanimotoSimilarity


def get_distance_matrix(mols):
    """
    Generate distance matrix from molecules using Morgan fingerprints.

    Args:
        mols: List of RDKit molecule objects

    Returns:
        torch.Tensor: Distance matrix (1 - Tanimoto similarity)
    """
    nvmol_gen = nvmolMorganGen(radius=2, fpSize=1024)
    nvmol_fps = nvmol_gen.GetFingerprints(mols, 10)
    sim_matrix = crossTanimotoSimilarity(nvmol_fps).torch()
    return 1.0 - sim_matrix


def resize_and_fill(distance_mat: torch.Tensor, want_size):
    """
    Resize distance matrix to desired size, filling with random values if needed.

    Args:
        distance_mat: Original distance matrix
        want_size: Desired size for the output matrix

    Returns:
        torch.Tensor: Resized distance matrix
    """
    current_size = distance_mat.shape[0]
    if current_size >= want_size:
        return distance_mat[:want_size, :want_size].contiguous()

    # Create larger matrix with random distances
    full_mat = torch.rand(want_size, want_size, dtype=distance_mat.dtype, device=distance_mat.device)
    # Make symmetric and ensure values are in reasonable range
    full_mat = torch.abs(full_mat - full_mat.T).clip(0.01, 0.99)
    full_mat.fill_diagonal_(0.0)
    # Copy original data
    full_mat[:current_size, :current_size] = distance_mat
    return full_mat


def main():
    """Main profiling function with NVTX annotations."""
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    size = int(sys.argv[1])
    cutoff = float(sys.argv[2])
    do_warmup = False
    if len(sys.argv) > 3:
        do_warmup = bool(int(sys.argv[3]))

    # Default input data path - adjust as needed
    input_data = "benchmarks/data/chembl_10k.smi"
    if len(sys.argv) > 4:
        input_data = sys.argv[4]

    with nvtx.annotate("Setup", color="blue"):
        print(f"Loading molecules from {input_data}")
        with open(input_data, "r") as f:
            smis = [line.strip() for line in f.readlines()]

        print(f"Parsing {size + 100} SMILES strings")
        mols = [MolFromSmiles(smi, sanitize=True) for smi in smis[: size + 100]]
        mols = [mol for mol in mols if mol is not None][:size]
        print(f"Successfully parsed {len(mols)} molecules")

        print("Generating distance matrix")
        dists = get_distance_matrix(mols)
        dist_mat = resize_and_fill(dists, size)
        print(f"Distance matrix shape: {dist_mat.shape}")

    if do_warmup:
        with nvtx.annotate("Warmup", color="red"):
            print("Running warmup with 10 molecules")
            warmup_result = butina_nvmol(dist_mat[:10, :10].contiguous(), 0.2)
            warmup_clusters = warmup_result.torch()
            num_warmup_clusters = len(torch.unique(warmup_clusters))
            print(f"Warmup completed, found {num_warmup_clusters} clusters")

    with nvtx.annotate("Clustering", color="green"):
        print(f"Running Butina clustering: size={size}, cutoff={cutoff}")
        result = butina_nvmol(dist_mat, cutoff)
        print("Clustering completed")

    # Print cluster size distribution
    clusters = result.torch()
    unique_clusters, counts = torch.unique(clusters, return_counts=True)
    num_clusters = len(unique_clusters)
    cluster_sizes = counts.cpu().tolist()

    print(f"\nCluster statistics:")
    print(f"  Total clusters: {num_clusters}")
    print(f"  Largest cluster: {max(cluster_sizes)}")
    print(f"  Smallest cluster: {min(cluster_sizes)}")
    print(f"  Average cluster size: {sum(cluster_sizes) / num_clusters:.2f}")
    print(f"  Singletons: {sum(1 for s in cluster_sizes if s == 1)}")


if __name__ == "__main__":
    main()
