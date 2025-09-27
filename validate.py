import argparse
from pathlib import Path

import numpy as np
from sklearn import metrics

DEFAULT_GT: Path = Path("test/gt.txt")
DEFAULT_LABELS: Path = Path("test/labels.txt")


def compute_stats(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict:
    """Compute clustering stats and NMI.

    Args:
        labels_true: Ground-truth labels (n,).
        labels_pred: Predicted labels (n,), with -1 for noise (optional).

    Returns:
        dict with keys:
            n: Number of points.
            nmi: Normalized mutual information (float).
            n_clusters: Number of clusters (excludes -1).
            n_noise: Number of noise points (label == -1).
            ratio_points_clustered: Fraction with label >= 0.
            cluster_sizes: dict[label -> count], excludes noise.
    """
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError(
            f"ground truth and labels have different lengths: "
            f"{labels_true.shape[0]} vs {labels_pred.shape[0]}"
        )

    n = labels_true.shape[0]

    # NMI (arithmetic average)
    nmi = metrics.normalized_mutual_info_score(
        labels_true, labels_pred, average_method="arithmetic"
    )

    # Basic counts
    clustered_mask = labels_pred >= 0
    ratio_points_clustered = float(clustered_mask.sum()) / float(n)
    n_noise = int(np.count_nonzero(labels_pred == -1))

    # Cluster labels excluding noise
    cluster_labels = np.unique(labels_pred[clustered_mask])
    n_clusters = int(cluster_labels.size)

    # Cluster sizes (exclude noise)
    cluster_sizes: dict[int, int] = {}
    if n_clusters > 0:
        # Using bincount over remapped labels for speed is possible, but clarity first
        for lab in cluster_labels:
            cluster_sizes[int(lab)] = int(np.count_nonzero(labels_pred == lab))

    return {
        "n": n,
        "nmi": float(nmi),
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
    print(f"number of clusters: {stats['n_clusters']}")
    print(f"number of noise: {stats['n_noise']}")
    print(f"ratio of points clustered: {stats['ratio_points_clustered']:.3f}")
    print(f"NMI value: {stats['nmi']:.3f}")

    if stats["cluster_sizes"]:
        print("\ncluster sizes:")
        for lab in sorted(stats["cluster_sizes"]):
            print(f"  {lab}: {stats['cluster_sizes'][lab]}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Compute NMI and simple clustering stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GT,
        help="Path to ground-truth labels.",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Path to labels to evaluate.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    y_true = np.loadtxt(args.ground_truth, dtype=int)
    y_pred = np.loadtxt(args.labels, dtype=int)
    stats = compute_stats(y_true, y_pred)
    print_stats(stats)


if __name__ == "__main__":
    main()
