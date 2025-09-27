import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

CWD = Path.cwd()
INPUT_DIR = CWD / "test" / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GT = INPUT_DIR / "gt.txt"
DEFAULT_LABELS = INPUT_DIR / "labels.txt"
DEFAULT_POINTS = INPUT_DIR / "points.txt"
DEFAULT_OUT_PNG = INPUT_DIR / "clusters.png"


def load_points(path: Path) -> np.ndarray:
    """Load points with leading ID column.

    Assumes each line is: `id x0 x1 ...`.

    Args:
        path: File path.

    Returns:
        Array of shape (n_samples, d) with coordinates only (IDs dropped).
    """
    raw = np.loadtxt(path, dtype=float)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] < 2:
        raise ValueError("points file must have at least id and one coordinate.")
    return raw[:, 1:]  # drop id


def compute_stats(
    labels_true: np.ndarray, labels_pred: np.ndarray, X: Optional[np.ndarray] = None
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
    silhouette_pred: Optional[float] = None
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
) -> Optional[Path]:
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Compute clustering indices and (if 2D) plot GT vs predicted.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ground-truth", type=Path, default=DEFAULT_GT, help="GT labels.")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="Pred labels.")
    p.add_argument("--points", type=Path, default=DEFAULT_POINTS, help="Points file.")
    p.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG, help="Plot path.")
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    X = load_points(args.points)
    y_true = np.loadtxt(args.ground_truth, dtype=int)
    y_pred = np.loadtxt(args.labels, dtype=int)

    stats = compute_stats(y_true, y_pred, X=X)
    print_stats(stats)

    out = plot_clusters_2d(X, y_true, y_pred, args.out_png)
    if out:
        print(f"wrote plot: {out}")


if __name__ == "__main__":
    main()
