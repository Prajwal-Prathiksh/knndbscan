import time
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from cuml.cluster import HDBSCAN as cuHDBSCAN
from cuml.cluster import KMeans as cuKMeans
from sklearn.datasets import make_blobs, make_moons

from knndbscan import knn_dbscan


def run_benchmark(
    X_cpu,
    dataset_name: str,
    n_clusters: int,
    eps: float,
    min_samples: int,
    seed=0,
    figs_dir: Path | None = None,
):
    print(f"\n{'=' * 20} Benchmarking on {dataset_name} (GPU) {'=' * 20}")
    print(f"Data shape: ({X_cpu.shape[0]:,}, {X_cpu.shape[1]})")

    # Move data to GPU
    start_transfer = time.perf_counter()
    X_gpu = cp.asarray(X_cpu)
    transfer_time = time.perf_counter() - start_transfer
    print(f"Data transfer to GPU: {transfer_time:.4f}s")

    results = []
    labels_dict = {}

    # --- cuKMeans ---
    print(f"Running cuKMeans (k={n_clusters})...")
    start = time.perf_counter()
    kmeans = cuKMeans(n_clusters=n_clusters, random_state=seed, init="k-means||")
    kmeans.fit(X_gpu)
    labels_kmeans = kmeans.labels_
    duration_kmeans = time.perf_counter() - start

    # Convert labels back to CPU for analysis/plotting
    labels_kmeans_cpu = cp.asnumpy(labels_kmeans)
    n_clus_km = len(np.unique(labels_kmeans_cpu))
    noise_km = 0

    results.append(
        {
            "Algorithm": "cuKMeans",
            "Time (s)": duration_kmeans,
            "Clusters": n_clus_km,
            "Noise Points": noise_km,
        }
    )
    labels_dict["cuKMeans"] = labels_kmeans_cpu

    # --- cuHDBSCAN ---
    # Note: Using HDBSCAN as requested to approximate DBSCAN behavior on GPU
    # adjust min_cluster_size/min_samples to match DBSCAN params roughly if needed,
    # but here we use the passed min_samples.
    print(f"Running cuHDBSCAN [min_cluster_size={min_samples}]...")
    start = time.perf_counter()
    # HDBSCAN doesn't use eps exactly like DBSCAN, but we can't easily map it perfectly.
    # We will use min_cluster_size = min_samples.
    hdbscan = cuHDBSCAN(min_cluster_size=min_samples, min_samples=min_samples)
    labels_hdbscan = hdbscan.fit_predict(X_gpu)
    duration_hdbscan = time.perf_counter() - start

    labels_hdbscan_cpu = cp.asnumpy(labels_hdbscan)
    n_clus_hdb = len(set(labels_hdbscan_cpu)) - (1 if -1 in labels_hdbscan_cpu else 0)
    noise_hdb = list(labels_hdbscan_cpu).count(-1)

    results.append(
        {
            "Algorithm": "cuHDBSCAN",
            "Time (s)": duration_hdbscan,
            "Clusters": n_clus_hdb,
            "Noise Points": noise_hdb,
        }
    )
    labels_dict["cuHDBSCAN"] = labels_hdbscan_cpu

    # --- knndbscan (GPU Mode) ---
    print(f"Running knndbscan [eps={eps}, min_samples={min_samples}, device='gpu']...")
    start = time.perf_counter()
    # Pass GPU array directly
    labels_knn_gpu = knn_dbscan(X_gpu, eps=eps, min_samples=min_samples, device="gpu")
    duration_knn = time.perf_counter() - start

    labels_knn_cpu = cp.asnumpy(labels_knn_gpu)
    n_clus_knn = len(set(labels_knn_cpu)) - (1 if -1 in labels_knn_cpu else 0)
    noise_knn = list(labels_knn_cpu).count(-1)

    results.append(
        {
            "Algorithm": "knndbscan (GPU)",
            "Time (s)": duration_knn,
            "Clusters": n_clus_knn,
            "Noise Points": noise_knn,
        }
    )
    labels_dict["knndbscan"] = labels_knn_cpu

    # --- Print summary table ---
    print("\nBenchmark Results (GPU):")
    print(f"{'Algorithm':<20} {'Time (s)':>10} {'Clusters':>10} {'Noise Points':>15}")
    print("-" * 60)
    for res in results:
        print(
            f"{res['Algorithm']:<20} {res['Time (s)']:>10.4f} {res['Clusters']:>10} {res['Noise Points']:>15}"
        )

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # cuKMeans
    axes[0].scatter(
        X_cpu[:, 0], X_cpu[:, 1], c=labels_dict["cuKMeans"], cmap="viridis", s=10
    )
    axes[0].set_title(f"cuKMeans (time: {duration_kmeans:.4f}s)")

    # cuHDBSCAN
    axes[1].scatter(
        X_cpu[:, 0], X_cpu[:, 1], c=labels_dict["cuHDBSCAN"], cmap="viridis", s=10
    )
    axes[1].set_title(f"cuHDBSCAN (time: {duration_hdbscan:.4f}s)")

    # knndbscan
    axes[2].scatter(
        X_cpu[:, 0], X_cpu[:, 1], c=labels_dict["knndbscan"], cmap="viridis", s=10
    )
    axes[2].set_title(f"knndbscan (GPU) (time: {duration_knn:.4f}s)")

    plt.suptitle(f"GPU Clustering Comparison: {dataset_name}")
    plt.tight_layout()

    fig_base_name = f"comparison_gpu_{dataset_name.lower().replace(' ', '_')}.png"
    if isinstance(figs_dir, Path):
        figs_dir.mkdir(parents=True, exist_ok=True)
        fig_fname = figs_dir / fig_base_name
    else:
        fig_fname = fig_base_name

    plt.savefig(fig_fname)
    print(f"\nPlot saved to {fig_fname}")
    plt.close()


if __name__ == "__main__":
    CWD = Path.cwd()
    FIGS_DIR = CWD / "figures"

    n_samples = 100_000
    seed = 0

    # 1. Blobs
    print("Generating Blobs...")
    n_blobs = 5
    X_blobs, _ = make_blobs(  # type: ignore
        n_samples=n_samples,
        centers=n_blobs,
        cluster_std=0.2,
        random_state=seed,
    )
    run_benchmark(
        X_blobs,
        dataset_name="Blobs",
        n_clusters=n_blobs,
        eps=0.5,
        min_samples=10,
        seed=seed,
        figs_dir=FIGS_DIR,
    )

    # 2. Moons
    print("Generating Moons...")
    X_moons, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    run_benchmark(
        X_moons,
        dataset_name="Moons",
        n_clusters=2,
        eps=0.2,
        min_samples=10,
        seed=seed,
        figs_dir=FIGS_DIR,
    )
