"""Compare kNN-DBSCAN with scikit-learn's DBSCAN on two 2-D datasets: time, clusters, agreement.

Run with `uv run python examples/compare_with_dbscan.py`; the plot goes to figures/.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from knndbscan import knn_dbscan

FIGURE = Path(__file__).resolve().parents[1] / "figures" / "compare_with_dbscan.png"
N_SAMPLES = 20_000
DATASETS = {
    # name: (data, eps, min_samples)
    "blobs": (
        make_blobs(N_SAMPLES, centers=5, cluster_std=0.6, random_state=0)[0],
        0.3,
        10,
    ),
    "moons": (make_moons(N_SAMPLES, noise=0.06, random_state=0)[0], 0.05, 15),
}


def timed(run, *args, **kwargs):
    start = time.perf_counter()
    labels = run(*args, **kwargs)
    return labels, time.perf_counter() - start


def describe(labels: np.ndarray) -> str:
    return f"{len(set(labels.tolist()) - {-1})} clusters, {(labels == -1).sum()} noise"


def main() -> None:
    figure, axes = plt.subplots(len(DATASETS), 2, figsize=(10, 4.5 * len(DATASETS)))
    for row, (name, (X, eps, min_samples)) in zip(axes, DATASETS.items(), strict=True):
        reference, t_reference = timed(
            DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict, X
        )
        labels, t_knn = timed(knn_dbscan, X, eps, min_samples, n_jobs=-1, device="cpu")
        print(f"{name} (eps={eps}, min_samples={min_samples})")
        print(f"  DBSCAN      {t_reference:7.3f} s  {describe(reference)}")
        print(f"  kNN-DBSCAN  {t_knn:7.3f} s  {describe(labels)}")
        print(
            f"  adjusted Rand index between them: {adjusted_rand_score(reference, labels):.4f}"
        )
        for axis, title, result in zip(
            row, ("DBSCAN", "kNN-DBSCAN"), (reference, labels), strict=True
        ):
            axis.scatter(X[:, 0], X[:, 1], c=result, s=1, cmap="tab10")
            axis.set_title(f"{name}: {title} ({describe(result)})")
            axis.set_aspect("equal")
    FIGURE.parent.mkdir(exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=150)
    print(f"plot saved to {FIGURE}")


if __name__ == "__main__":
    main()
