from pathlib import Path

import numpy as np

from knndbscan import _core
from validate import compute_stats, plot_clusters_2d, print_stats


def load_data():
    """Load points and ground truth labels from files."""
    X = np.loadtxt("test/input/points.txt")
    y_true = np.loadtxt("test/input/gt.txt", dtype=int)
    return X, y_true


def load_knn_graph():
    """Load k-NN graph from input.txt, parsing only the first 19 neighbors per point."""
    N = 40000
    k = 19  # minPts - 1
    JA = []
    A = []
    with open("test/input/input.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            # Skip gid, dist_1st, id_1st; take first 19 neighbor pairs
            neighbors = parts[3:]
            for i in range(0, 38, 2):  # 19 pairs * 2 = 38 elements
                dist = float(neighbors[i])
                idx = int(neighbors[i + 1])
                A.append(dist)
                JA.append(idx)
    JA = np.array(JA, dtype=np.int32)
    A = np.array(A, dtype=np.float32)
    return N, k, JA, A


def main():
    # Load data
    X, y_true = load_data()
    N, k, JA, A = load_knn_graph()

    # Parameters
    eps = 1300.0
    minPts = 20

    # Run clustering
    labels_pred = _core.knndbscan(N, eps, minPts, k, JA, A)

    # Compute and print stats
    stats = compute_stats(y_true, labels_pred, X=X)
    print_stats(stats)

    # Plot results
    out_png = Path("test/input/clusters_binding.png")
    plot_path = plot_clusters_2d(X, y_true, labels_pred, out_png)
    if plot_path:
        print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
