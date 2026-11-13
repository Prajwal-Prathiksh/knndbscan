"""Thread scaling of the clustering step alone, on a synthetic kNN graph of a few million points.

Run with `uv run python examples/benchmark_threads.py [n_points]`. The graph is random,
so neighbor search does not enter the timings; labels must be identical at every thread
count.
"""

import os
import sys
import time

import numpy as np

from knndbscan import knn_dbscan_from_graph

N_NEIGHBORS = 19
MIN_SAMPLES = N_NEIGHBORS + 1
EPS = 95.0  # about 0.95**19 = 38% core points
REPEATS = 3


def synthetic_graph(n_points: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random neighbors with sorted uniform distances; self-loops become the next point."""
    rng = np.random.default_rng(seed)
    neighbors = rng.integers(0, n_points, size=(n_points, N_NEIGHBORS), dtype=np.int32)
    rows = np.arange(n_points, dtype=np.int32)[:, None]
    neighbors = np.where(neighbors == rows, (rows + 1) % n_points, neighbors)
    distances = np.sort(
        rng.uniform(0, 100, size=neighbors.shape).astype(np.float32), axis=1
    )
    return neighbors, distances


def main() -> None:
    n_points = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    neighbors, distances = synthetic_graph(n_points)
    n_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    thread_counts = sorted({1, *(2**i for i in range(1, 8) if 2**i <= n_cpus), n_cpus})

    print(
        f"{n_points:,} points, {N_NEIGHBORS} neighbors, eps={EPS}, min_samples={MIN_SAMPLES}"
    )
    print(f"{'threads':>8} {'time (s)':>10} {'speedup':>8}")
    reference, baseline = None, None
    for n_jobs in thread_counts:
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            labels = knn_dbscan_from_graph(
                neighbors, distances, EPS, MIN_SAMPLES, n_jobs=n_jobs
            )
            times.append(time.perf_counter() - start)
        best = min(times)
        baseline = baseline or best
        if reference is None:
            reference = labels
        assert np.array_equal(labels, reference), "labels changed with the thread count"
        print(f"{n_jobs:>8} {best:>10.3f} {baseline / best:>7.2f}x")
    print(
        f"{len(set(reference.tolist()) - {-1})} clusters, {(reference == -1).sum():,} noise points"
    )


if __name__ == "__main__":
    main()
