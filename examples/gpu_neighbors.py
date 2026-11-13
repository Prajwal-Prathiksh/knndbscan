"""Neighbor search on the GPU (cuML) against the CPU (scikit-learn), on the same data.

Needs the `gpu` extra and a CUDA device: `uv sync --extra gpu`, then
`uv run python examples/gpu_neighbors.py`. Clustering runs on the CPU in both cases.
"""

import sys
import time

from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from knndbscan import knn_dbscan
from knndbscan.neighbors import gpu_available

EPS = 2.5
MIN_SAMPLES = 20


def main() -> None:
    if not gpu_available():
        sys.exit(
            "No usable GPU: install the gpu extra and check that CUDA sees a device."
        )
    X, _ = make_blobs(n_samples=200_000, n_features=32, centers=8, random_state=0)
    results = {}
    for device in ("cpu", "gpu"):
        start = time.perf_counter()
        results[device] = knn_dbscan(X, EPS, MIN_SAMPLES, n_jobs=-1, device=device)
        labels = results[device]
        print(
            f"{device}: {time.perf_counter() - start:7.2f} s, "
            f"{len(set(labels.tolist()) - {-1})} clusters, {(labels == -1).sum():,} noise"
        )
    # cuML computes distances in float32, so points at distance ~eps can flip.
    print(
        f"adjusted Rand index cpu vs gpu: {adjusted_rand_score(results['cpu'], results['gpu']):.6f}"
    )


if __name__ == "__main__":
    main()
