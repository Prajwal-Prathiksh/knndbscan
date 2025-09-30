import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors

from knndbscan import knndbscan
from validate import compute_stats, plot_clusters_2d, print_stats

# Constants
N_SAMPLES = 10_000
NOISE = 0.05
RANDOM_STATE = 42
EPS = 1300.0
MIN_PTS = 20
THREADS = 16
OUT_PNG = Path("test/input/clusters_binding.png")


def make_knn_graph(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a k-NN graph over X.

    Args:
        X: Array of shape (n_samples, n_features).
        k: Number of neighbors to return (includes self).

    Returns:
        distances: Array of shape (n_samples, k) with Euclidean distance50
        indices: Array of shape (n_samples, k) with neighbor indices.
    """
    n_samples = X.shape[0]
    k = min(k, n_samples)

    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    t1 = time.perf_counter()
    print(
        f"Built k-NN graph with k={k} for {n_samples} samples - shape: {distances.shape} in {t1 - t0:.2f} seconds"
    )
    return distances, indices


def load_data():
    """Generate points and ground truth labels using make_moons."""
    X, y_true = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=RANDOM_STATE)
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} features.")
    return X, y_true


def load_knn_graph(X):
    N = N_SAMPLES
    minPts = MIN_PTS
    k = minPts - 1
    graph_k = minPts  # includes self
    distances, indices = make_knn_graph(X, graph_k)

    # Skip self (first neighbor)
    A = distances[:, 1:].flatten().astype(np.float32)
    JA = indices[:, 1:].flatten().astype(np.int32)

    # Print memory usage
    mem_JA = JA.nbytes / (1024**2)
    mem_A = A.nbytes / (1024**2)
    total_mem = mem_JA + mem_A
    print(
        f"Memory usage for k-NN graph: JA: {mem_JA:.2f} MB, A: {mem_A:.2f} MB, Total: {total_mem:.2f} MB"
    )

    return N, k, JA, A


def main():
    # Load data
    X, y_true = load_data()
    N, k, JA, A = load_knn_graph(X)

    # Parameters
    eps = EPS
    minPts = MIN_PTS

    # Run clustering
    t1 = time.perf_counter()
    labels_pred = knndbscan(N, eps, minPts, k, JA, A, threads=THREADS)
    t2 = time.perf_counter()
    print(f"Clustering took {t2 - t1:.2f} seconds")

    # Compute and print stats
    stats = compute_stats(y_true, labels_pred, X=X)
    print_stats(stats)

    # Plot results
    out_png = OUT_PNG
    plot_path = plot_clusters_2d(X, y_true, labels_pred, out_png)
    if plot_path:
        print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
