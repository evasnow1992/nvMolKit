import torch
import numpy as np
import time
import statistics


def tanimoto_similarity(fps1, fps2):
    """Compute Tanimoto similarity between two sets of fingerprints.

    Args:
        fps1: Binary fingerprint tensor of shape (n1, bits)
        fps2: Binary fingerprint tensor of shape (n2, bits)

    Returns:
        Similarity matrix of shape (n1, n2) with Tanimoto similarities
    """
    similarities = torch.zeros((fps1.shape[0], fps2.shape[0]), device=fps1.device, dtype=torch.float64)
    b_sums = fps2.sum(axis=1)

    for i in range(fps1.shape[0]):
        fp1_section = fps1[i, :]
        fp_and_sum = (fp1_section & fps2).sum(axis=1)
        similarities[i, :] = fp_and_sum / (fp1_section.sum() + b_sums - fp_and_sum)

    return similarities


def tanimoto_similarity_vectorized(fps1, fps2, chunk_size=1000):
    """Vectorized version of Tanimoto similarity computation with chunked processing.

    Args:
        fps1: Binary fingerprint tensor of shape (n1, bits)
        fps2: Binary fingerprint tensor of shape (n2, bits)
        chunk_size: Process in chunks to avoid memory issues

    Returns:
        Similarity matrix of shape (n1, n2) with Tanimoto similarities
    """
    n1, n2 = fps1.shape[0], fps2.shape[0]
    device = fps1.device

    # Pre-compute sums for fps2 (only once)
    fps2_sums = fps2.sum(dim=1)  # (n2,)

    # Initialize result tensor
    similarities = torch.zeros((n1, n2), device=device, dtype=torch.float32)

    # Process in chunks to avoid memory issues
    for i in range(0, n1, chunk_size):
        end_i = min(i + chunk_size, n1)
        fps1_chunk = fps1[i:end_i]  # (chunk_size, bits)

        for j in range(0, n2, chunk_size):
            end_j = min(j + chunk_size, n2)
            fps2_chunk = fps2[j:end_j]  # (chunk_size, bits)

            # Compute intersection for this chunk
            fps1_expanded = fps1_chunk.unsqueeze(1)  # (chunk_i, 1, bits)
            fps2_expanded = fps2_chunk.unsqueeze(0)  # (1, chunk_j, bits)
            intersection = (fps1_expanded & fps2_expanded).sum(dim=2)  # (chunk_i, chunk_j)

            # Compute union for this chunk
            fps1_sums_chunk = fps1_chunk.sum(dim=1, keepdim=True)  # (chunk_i, 1)
            fps2_sums_chunk = fps2_sums[j:end_j]  # (chunk_j,)
            union = fps1_sums_chunk + fps2_sums_chunk - intersection  # (chunk_i, chunk_j)

            # Compute similarities for this chunk
            similarities[i:end_i, j:end_j] = intersection.float() / union.float()

    return similarities


def get_optimal_chunk_size(n, bits, device):
    """Determine optimal chunk size based on available GPU memory.

    Args:
        n: Number of fingerprints
        bits: Number of bits per fingerprint
        device: Device ('cuda' or 'cpu')

    Returns:
        Optimal chunk size
    """
    if device == "cpu":
        # CPU can handle larger chunks, but still limit to avoid excessive memory usage
        return min(n, 2000)

    try:
        # Get available GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        available_memory = total_memory - allocated_memory

        # Reserve some memory for other operations (use 70% of available)
        usable_memory = int(available_memory * 0.7)

        # Estimate memory needed for chunk_size x chunk_size x bits tensor
        # Each element is 1 byte (bool), plus some overhead
        bytes_per_element = 4  # Conservative estimate including intermediate tensors

        # Memory needed: chunk_size^2 * bits * bytes_per_element
        max_chunk_elements = usable_memory // (bits * bytes_per_element)
        max_chunk_size = int(max_chunk_elements**0.5)  # sqrt for square chunks

        # Clamp to reasonable bounds
        chunk_size = max(100, min(max_chunk_size, n, 2000))

        print(f"Auto-detected chunk size: {chunk_size} (available GPU memory: {available_memory / 1e9:.1f} GB)")
        return chunk_size

    except Exception as e:
        print(f"Could not determine optimal chunk size: {e}")
        return min(n, 1000)  # Conservative fallback


def generate_random_fingerprints(n, bits=1024, density=0.1, device="cuda"):
    """Generate random binary fingerprints.

    Args:
        n: Number of fingerprints
        bits: Number of bits per fingerprint
        density: Fraction of bits set to 1
        device: Device to create tensors on

    Returns:
        Binary tensor of shape (n, bits)
    """
    # Generate random fingerprints with specified density
    fps = torch.rand(n, bits, device=device) < density
    return fps.bool()


def benchmark_tanimoto(n=16000, bits=1024, num_runs=5, use_vectorized=True):
    """Benchmark Tanimoto similarity computation.

    Args:
        n: Number of fingerprints (will compute n x n similarities)
        bits: Number of bits per fingerprint
        num_runs: Number of benchmark runs
        use_vectorized: Whether to use vectorized implementation

    Returns:
        tuple: (mean_time, std_error, total_comparisons)
    """
    print("Benchmarking PyTorch Tanimoto similarity...")
    print(f"Parameters: n={n}, bits={bits}, runs={num_runs}")
    print(f"Using {'vectorized' if use_vectorized else 'loop-based'} implementation")

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("Warning: CUDA not available, using CPU. Results may be much slower.")

    # Generate test data
    print("Generating random fingerprints...")
    fps = generate_random_fingerprints(n, bits, device=device)

    # Choose implementation and determine chunk size
    if use_vectorized:
        chunk_size = get_optimal_chunk_size(n, bits, device)
        tanimoto_func = lambda x, y: tanimoto_similarity_vectorized(x, y, chunk_size)
    else:
        tanimoto_func = tanimoto_similarity

    # Warm up GPU
    if device == "cuda":
        print("Warming up GPU...")
        _ = tanimoto_func(fps[:100], fps[:100])
        torch.cuda.synchronize()

    # Benchmark
    times = []
    print(f"Running {num_runs} benchmark iterations...")

    for run in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # Compute all-vs-all similarities
        _ = tanimoto_func(fps, fps)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times.append(elapsed)

        print(f"Run {run + 1}/{num_runs}: {elapsed:.4f} seconds")

    # Calculate statistics
    mean_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
    std_error = std_dev / np.sqrt(len(times))
    total_comparisons = n * n

    print("\nResults:")
    print(f"Mean time: {mean_time:.6f} ± {std_error:.6f} seconds")
    print(f"Total comparisons: {total_comparisons:,}")
    print(f"Throughput: {total_comparisons / mean_time:,.0f} comparisons/second")

    return mean_time, std_error, total_comparisons


def verify_correctness():
    """Verify that both implementations give the same results."""
    print("Verifying correctness of implementations...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Small test case
    n_test = 10
    fps_test = generate_random_fingerprints(n_test, bits=64, device=device)

    # Compute with both methods
    sim1 = tanimoto_similarity(fps_test, fps_test)
    sim2 = tanimoto_similarity_vectorized(fps_test, fps_test)

    # Check if results are close
    max_diff = torch.max(torch.abs(sim1 - sim2)).item()
    print(f"Maximum difference between implementations: {max_diff}")

    if max_diff < 1e-6:
        print("✓ Both implementations give identical results")
        return True
    else:
        print("✗ Implementations differ significantly!")
        return False


def main():
    """Main benchmarking function."""
    # Verify correctness first
    if not verify_correctness():
        print("Correctness check failed. Exiting.")
        return

    # Benchmark parameters matching your plot data
    n = 16000
    bits = 1024
    num_runs = 5

    print(f"Attempting to benchmark with n={n}...")

    try:
        # Try GPU first with chunked processing
        mean_time, std_error, total_comparisons = benchmark_tanimoto(
            n=n, bits=bits, num_runs=num_runs, use_vectorized=True
        )

        print("\n" + "=" * 60)
        print("FINAL RESULTS FOR PLOT:")
        print(f"n = {n}")
        print(f"Total comparisons = {total_comparisons:,}")
        print(f"Mean time = {mean_time:.6f} seconds")
        print(f"Standard error = {std_error:.6f} seconds")
        print(f"Throughput = {total_comparisons / mean_time:,.0f} comparisons/second")
        print("=" * 60)

    except torch.cuda.OutOfMemoryError as e:
        print(f"GPU out of memory error: {e}")
        print("Falling back to CPU or smaller problem size...")

        # Try smaller sizes to find what fits
        test_sizes = [8000, 4000, 2000, 1000]
        for test_n in test_sizes:
            try:
                print(f"\nTrying n={test_n}...")
                mean_time, std_error, total_comparisons = benchmark_tanimoto(
                    n=test_n, bits=bits, num_runs=3, use_vectorized=True
                )

                # Extrapolate to n=16000
                scaling_factor = (16000 / test_n) ** 2
                extrapolated_time = mean_time * scaling_factor
                extrapolated_error = std_error * scaling_factor

                print("\n" + "=" * 60)
                print("EXTRAPOLATED RESULTS FOR PLOT (n=16000):")
                print(f"Based on n={test_n} benchmark")
                print(f"Measured time for n={test_n}: {mean_time:.6f} ± {std_error:.6f} seconds")
                print(f"Extrapolated time for n=16000: {extrapolated_time:.6f} ± {extrapolated_error:.6f} seconds")
                print(f"Total comparisons = {16000 * 16000:,}")
                print(f"Extrapolated throughput = {(16000 * 16000) / extrapolated_time:,.0f} comparisons/second")
                print("=" * 60)
                break

            except Exception as inner_e:
                print(f"n={test_n} also failed: {inner_e}")
                continue

        else:
            print("All GPU attempts failed. Trying CPU...")
            try:
                # Force CPU usage
                device_backup = torch.cuda.is_available
                torch.cuda.is_available = lambda: False

                mean_time, std_error, total_comparisons = benchmark_tanimoto(
                    n=min(n, 2000), bits=bits, num_runs=2, use_vectorized=False
                )

                torch.cuda.is_available = device_backup
                print("CPU benchmark completed successfully")

            except Exception as cpu_e:
                print(f"CPU benchmark also failed: {cpu_e}")

    # Also test loop-based version for comparison (if we haven't already)
    print("\nFor comparison, testing loop-based implementation...")
    try:
        test_n_loop = min(n, 1000)  # Use smaller n for loop version as it's much slower
        mean_time_loop, std_error_loop, _ = benchmark_tanimoto(
            n=test_n_loop,
            bits=bits,
            num_runs=3,
            use_vectorized=False,
        )

        if n > test_n_loop:
            # Extrapolate to full size
            scaling_factor_loop = (n / test_n_loop) ** 2
            extrapolated_time_loop = mean_time_loop * scaling_factor_loop
            extrapolated_error_loop = std_error_loop * scaling_factor_loop

            print("\n" + "-" * 60)
            print("LOOP-BASED EXTRAPOLATED RESULTS:")
            print(f"Based on n={test_n_loop} benchmark")
            print(f"Measured time for n={test_n_loop}: {mean_time_loop:.6f} ± {std_error_loop:.6f} seconds")
            print(f"Extrapolated time for n={n}: {extrapolated_time_loop:.6f} ± {extrapolated_error_loop:.6f} seconds")
            print(f"Extrapolated throughput = {(n * n) / extrapolated_time_loop:,.0f} comparisons/second")
            print("-" * 60)
        else:
            print(f"Loop-based results for n={test_n_loop}:")
            print(f"Time: {mean_time_loop:.6f} ± {std_error_loop:.6f} seconds")
            print(f"Throughput: {(test_n_loop * test_n_loop) / mean_time_loop:,.0f} comparisons/second")

    except Exception as loop_e:
        print(f"Loop-based benchmark failed: {loop_e}")


if __name__ == "__main__":
    main()
