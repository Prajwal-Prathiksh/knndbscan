#!/usr/bin/env python3
"""
Threading Benchmark for k-NN DBSCAN core (knndbscan.knndbscan).

- Generates a deterministic synthetic k-NN graph (exact shape you need), so that
  k-NN build time doesn't dominate and you can focus on the clustering kernel.
- Benchmarks multiple thread counts with repeated runs.
- Prints a neat summary and saves pretty plots (PNG/PDF).
"""


import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from knndbscan import knndbscan

# =========================
# ======= CONSTANTS =======
# =========================
# Data / graph
N: int = 40_000_000
EPS: float = 100.0
MIN_PTS: int = (
    20  # includes self if you were to keep it; we drop one column -> k = MIN_PTS - 1
)
RNG_SEED: int = 42
DIST_RANGE: tuple[float, float] = (0.0, 200.0)  # range for synthetic distances

# Benchmarking
THREAD_COUNTS: list[int] = [1, 2, 4, 6, 8, 10, 12, 14, 16]
N_RUNS_PER_SETTING: int = 3  # repetitions for each thread count

# Output
OUT_BASENAME: str = "threading_performance"

# Plot look
FIGSIZE: tuple[int, int] = (14, 5)
DPI: int = 300
GRID_ALPHA: float = 0.25
MARKER_SIZE: int = 7
LINE_WIDTH: int = 2


# =========================
# ===== DATA CLASSES ======
# =========================
@dataclass
class BenchResult:
    time_avg: float
    time_std: float
    clusters: int


# =========================
# ====== UTILITIES ========
# =========================
def _ensure_no_self_neighbors(indices: np.ndarray, n: int) -> np.ndarray:
    """
    Replace any self-neighbor (indices[i, j] == i) with (i+1) % n.
    Works vectorized without Python loops.
    """
    ar = np.arange(n, dtype=indices.dtype)[:, None]  # shape (n,1)
    same = indices == ar  # (n, k)
    # next index per row:
    next_idx = (ar + 1) % n  # (n,1)
    # Broadcast next_idx to indices shape, then replace where same
    return np.where(same, np.broadcast_to(next_idx, indices.shape), indices)


def _sort_neighbors_by_distance(
    dist: np.ndarray, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort neighbors within each row by ascending distance.
    Returns sorted (dist, idx).
    """
    order = np.argsort(dist, axis=1)
    rows = np.arange(dist.shape[0])[:, None]
    dist_sorted = dist[rows, order]
    idx_sorted = idx[rows, order]
    return dist_sorted, idx_sorted


def generate_synthetic_knn_graph(
    n: int,
    min_pts: int,
    dist_range: tuple[float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a deterministic synthetic k-NN list (distances, indices), then
    sort neighbors by distance and ensure no self-neighbors.
    """
    rng = np.random.default_rng(seed)
    # Uniform distances in [lo, hi)
    distances = rng.uniform(dist_range[0], dist_range[1], size=(n, min_pts)).astype(
        np.float32
    )
    indices = rng.integers(0, n, size=(n, min_pts), dtype=np.int32)

    indices = _ensure_no_self_neighbors(indices, n)
    distances, indices = _sort_neighbors_by_distance(distances, indices)
    return distances, indices


def flatten_graph(
    distances: np.ndarray, indices: np.ndarray, min_pts: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Drop the first neighbor column (often self or closest neighbor)
    and flatten to CSR-like arrays A (weights) and JA (indices) expected by k-NN DBSCAN.
    """
    A = distances[:, 1:].reshape(-1).astype(np.float32, copy=False)
    JA = indices[:, 1:].reshape(-1).astype(np.int32, copy=False)
    k = min_pts - 1
    return A, JA, k


def run_knndbscan_once(
    n: int,
    eps: float,
    min_pts: int,
    k: int,
    JA: np.ndarray,
    A: np.ndarray,
    threads: int,
) -> tuple[np.ndarray, float]:
    """
    Run the core DBSCAN once and return (labels, elapsed_time_seconds).
    """
    start = time.perf_counter()
    labels = knndbscan(n, eps, min_pts, k, JA, A, threads=threads)
    elapsed = time.perf_counter() - start
    return labels, elapsed


def summarize_labels(labels: np.ndarray) -> int:
    """Return number of clusters excluding noise (-1)."""
    if labels.size == 0:
        return 0
    unique = np.unique(labels)
    n_clusters = unique.size - (1 if -1 in unique else 0)
    return int(n_clusters)


def benchmark_over_threads(
    n: int,
    eps: float,
    min_pts: int,
    k: int,
    JA: np.ndarray,
    A: np.ndarray,
    thread_counts: list[int],
    n_runs: int,
) -> dict[int, BenchResult]:
    """
    For each thread count, run the kernel n_runs times, average time, compute std, and record clusters.
    """
    results: dict[int, BenchResult] = {}
    for t in thread_counts:
        times: list[float] = []
        last_labels = None
        for _ in range(n_runs):
            labels, dt = run_knndbscan_once(n, eps, min_pts, k, JA, A, t)
            times.append(dt)
            last_labels = labels  # we only need cluster count once

        time_avg = float(np.mean(times))
        time_std = float(np.std(times))
        if last_labels is None:
            clusters = 0
        else:
            clusters = summarize_labels(last_labels)
        results[t] = BenchResult(time_avg, time_std, clusters)
    return results


def print_summary(results: dict[int, BenchResult]) -> None:
    """
    Pretty-prints a text summary table with times and speedups.
    """
    print("\nRESULTS SUMMARY")
    thread_list = sorted(results.keys())
    baseline = results[min(thread_list)].time_avg

    header = f"{'Threads':>8} | {'Time (s)':>10} | {'Std (s)':>9} | {'Clusters':>8} | {'Speedup':>8}"
    print(header)
    print("-" * len(header))
    for t in thread_list:
        r = results[t]
        speedup = baseline / r.time_avg if r.time_avg > 0 else float("inf")
        print(
            f"{t:8d} | {r.time_avg:10.3f} | {r.time_std:9.3f} | {r.clusters:8d} | {speedup:8.2f}"
        )

    max_speed = max((baseline / results[t].time_avg for t in thread_list), default=1.0)
    print("\nCONCLUSION:")
    if max_speed > 1.5:
        print(f"Good threading scalability achieved! Max speedup: {max_speed:.1f}x")
    else:
        print("Limited threading benefits observed. Possible reasons:")
        print("- Parallel regions not on the critical path")
        print(
            "- Synthetic graph not stressing parallel code paths similarly to real data"
        )
        print("- Memory bandwidth, NUMA effects, or oversubscription")


def make_plots(results: dict[int, BenchResult], out_basename: str) -> None:
    """
    Create two plots: runtime vs threads (with error bars) and speedup vs threads.
    Saves PNG/PDF according to flags.
    """
    thread_list = sorted(results.keys())
    times = np.array([results[t].time_avg for t in thread_list])
    stds = np.array([results[t].time_std for t in thread_list])
    baseline = times[0]
    speedups = baseline / times

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    # --- Plot 1: Runtime vs Threads
    ax = axes[0]
    ax.errorbar(
        thread_list,
        times,
        yerr=stds,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        capsize=5,
    )
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("k-NN DBSCAN Core: Runtime vs Threads")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xticks(thread_list)
    for t, y in zip(thread_list, times):
        ax.annotate(
            f"{y:.3f}s",
            xy=(t, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    # --- Plot 2: Speedup vs Threads
    ax = axes[1]
    ax.plot(
        thread_list,
        speedups,
        marker="s",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label="Measured",
    )
    ax.plot(
        thread_list,
        np.array(thread_list) / thread_list[0],
        linestyle="--",
        label="Ideal (linear)",
    )
    ax.axhline(1.0, linestyle=":", alpha=0.6, label="Baseline (1 thread)")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("k-NN DBSCAN Core: Speedup vs Threads")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_xticks(thread_list)
    ax.legend()
    for t, s in zip(thread_list, speedups):
        ax.annotate(
            f"{s:.2f}×",
            xy=(t, s),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    plt.tight_layout()

    plt.savefig(f"{out_basename}.png", dpi=DPI, bbox_inches="tight")


# =========================
# ========= MAIN ==========
# =========================
def main() -> None:
    print("Simple Threading Benchmark for k-NN DBSCAN Core")
    print("=" * 60)
    print(f"Dataset: {N:,} samples | eps={EPS} | minPts={MIN_PTS} | seed={RNG_SEED}")
    print(f"Threads to test: {THREAD_COUNTS}")
    print()

    # Generate deterministic synthetic neighbor lists
    print("Preparing synthetic k-NN graph ...")
    t0 = time.perf_counter()
    distances, indices = generate_synthetic_knn_graph(N, MIN_PTS, DIST_RANGE, RNG_SEED)
    A, JA, k = flatten_graph(distances, indices, MIN_PTS)
    t1 = time.perf_counter()
    mem_gb = (A.nbytes + JA.nbytes) / (1024**3)
    print(f"Graph prep: {(t1 - t0):.3f} s | Memory: {mem_gb:.2f} GB | k = {k}\n")

    # Run benchmark
    results = benchmark_over_threads(
        N, EPS, MIN_PTS, k, JA, A, THREAD_COUNTS, N_RUNS_PER_SETTING
    )

    # Summary + plots
    print_summary(results)
    make_plots(results, OUT_BASENAME)


if __name__ == "__main__":
    main()
