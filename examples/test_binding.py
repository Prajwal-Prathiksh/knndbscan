import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors

from knndbscan import run_knndbscan

ROOT_DIR = Path(__file__).parent.parent.resolve()
FIGURES_DIRECTORY = ROOT_DIR / "figures"
FIGURES_DIRECTORY.mkdir(exist_ok=True, parents=True)

# Constants
N_SAMPLES = 10_000
NOISE = 0.05
RANDOM_STATE = 42
EPS = 1300.0
MIN_PTS = 20
THREADS = 16
OUT_PNG = FIGURES_DIRECTORY / "clusters_binding.png"


# ====== UTILITIES ========
def compute_stats(
    labels_true: np.ndarray, labels_pred: np.ndarray, X: np.ndarray | None = None
) -> dict:
    """Compute clustering stats and indices.

    Args:
        labels_true: Ground-truth labels (n,).
        labels_pred: Predicted labels (n,), -1 treated as noise.
        X: Optional features (n, d) for silhouette on predicted labels.

    Returns:
        dict with keys:
            n, nmi, ari, v_measure, silhouette_pred,
            n_clusters, n_noise, ratio_points_clustered, cluster_sizes.
    """
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError(
            f"ground truth and labels have different lengths: "
            f"{labels_true.shape[0]} vs {labels_pred.shape[0]}"
        )

    n = int(labels_true.shape[0])

    nmi = float(
        metrics.normalized_mutual_info_score(
            labels_true, labels_pred, average_method="arithmetic"
        )
    )
    ari = float(metrics.adjusted_rand_score(labels_true, labels_pred))
    v_measure = float(metrics.v_measure_score(labels_true, labels_pred))

    # Basic counts
    clustered_mask = labels_pred >= 0
    ratio_points_clustered = float(clustered_mask.sum()) / float(n)
    n_noise = int(np.count_nonzero(labels_pred == -1))
    cluster_labels = np.unique(labels_pred[clustered_mask])
    n_clusters = int(cluster_labels.size)

    # Cluster sizes (exclude noise)
    cluster_sizes: dict[int, int] = {}
    for lab in cluster_labels:
        cluster_sizes[int(lab)] = int(np.count_nonzero(labels_pred == lab))

    # Silhouette for predicted labels (exclude noise; need >=2 clusters)
    silhouette_pred: float | None = None
    if X is not None and n_clusters >= 2:
        Xc = X[clustered_mask]
        yc = labels_pred[clustered_mask]
        # valid only if every cluster has at least 2 points
        _, counts = np.unique(yc, return_counts=True)
        if np.all(counts >= 2):
            try:
                silhouette_pred = float(
                    metrics.silhouette_score(Xc, yc, metric="euclidean")
                )
            except Exception:
                silhouette_pred = None

    return {
        "n": n,
        "nmi": nmi,
        "ari": ari,
        "v_measure": v_measure,
        "silhouette_pred": silhouette_pred,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "ratio_points_clustered": ratio_points_clustered,
        "cluster_sizes": cluster_sizes,
    }


def print_stats(stats: dict) -> None:
    """Pretty-print clustering stats.

    Args:
        stats: Output of compute_stats().
    """
    print(f"number of points: {stats['n']}")
    print(f"number of clusters: {stats['n_clusters']}")
    print(f"number of noise: {stats['n_noise']}")
    print(f"ratio of points clustered: {stats['ratio_points_clustered']:.3f}")
    print(f"NMI: {stats['nmi']:.4f}")
    print(f"Adjusted Rand Index: {stats['ari']:.4f}")
    print(f"V-measure: {stats['v_measure']:.4f}")
    if stats["silhouette_pred"] is None:
        print("Silhouette (pred): N/A")
    else:
        print(f"Silhouette (pred): {stats['silhouette_pred']:.4f}")

    if stats["cluster_sizes"]:
        print("\ncluster sizes (pred, excluding noise):")
        for lab in sorted(stats["cluster_sizes"]):
            print(f"  {lab}: {stats['cluster_sizes'][lab]}")


def plot_clusters_2d(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png: Path,
) -> Path | None:
    """Plot 2D GT vs predicted clusters side-by-side.

    Args:
        X: Data (n, 2).
        y_true: Ground truth (n,).
        y_pred: Predicted (n,), -1 is noise.
        out_png: Output path for the PNG.

    Returns:
        Path if written, else None.
    """
    if X.shape[1] != 2:
        print("plot skipped (data not 2D).")
        return None

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Left: ground truth
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, s=12)
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("x0")
    axes[0].set_ylabel("x1")

    # Right: predicted; noise marked distinctly
    noise_mask = y_pred == -1
    if np.any(~noise_mask):
        axes[1].scatter(
            X[~noise_mask, 0], X[~noise_mask, 1], c=y_pred[~noise_mask], s=12
        )
    if np.any(noise_mask):
        axes[1].scatter(X[noise_mask, 0], X[noise_mask, 1], marker="x", s=20)

    title = "Predicted"
    if np.any(noise_mask):
        title += " (noise = ×)"
    axes[1].set_title(title)
    axes[1].set_xlabel("x0")
    axes[1].set_ylabel("x1")

    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def make_knn_graph(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a k-NN graph over X.

    Args:
        X: Array of shape (n_samples, n_features).
        k: Number of neighbors to return (includes self).

    Returns:
        distances: Array of shape (n_samples, k) with Euclidean distances
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


# ======= MAIN ========
def main():
    # Load data
    X, y_true = load_data()
    N, k, JA, A = load_knn_graph(X)

    # Parameters
    eps = EPS
    minPts = MIN_PTS

    # Run clustering
    t1 = time.perf_counter()
    labels_pred = run_knndbscan(N, eps, minPts, k, JA, A, threads=THREADS)
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
