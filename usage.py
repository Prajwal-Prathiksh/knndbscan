import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons

from knndbscan import knndbscan


def run_benchmark(
    X,
    dataset_name,
    n_clusters,
    eps,
    min_samples,
    seed=0,
    figs_dir: Path | None = None,
):
    print(f"\n{'=' * 20} Benchmarking on {dataset_name} {'=' * 20}")
    print(f"Data shape: {X.shape}")

    results = []
    labels_dict = {}

    # --- KMeans ---
    print(f"Running KMeans (k={n_clusters})...")
    start = time.perf_counter()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels_kmeans = kmeans.fit_predict(X)
    duration_kmeans = time.perf_counter() - start

    n_clus_km = len(np.unique(labels_kmeans))
    noise_km = 0  # KMeans doesn't have noise

    results.append(
        {
            "Algorithm": "KMeans",
            "Time (s)": duration_kmeans,
            "Clusters": n_clus_km,
            "Noise Points": noise_km,
        }
    )
    labels_dict["KMeans"] = labels_kmeans

    # --- DBSCAN (sklearn) ---
    print(f"Running DBSCAN (sklearn) [eps={eps}, min_samples={min_samples}]...")
    start = time.perf_counter()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels_dbscan = dbscan.fit_predict(X)
    duration_dbscan = time.perf_counter() - start

    n_clus_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    noise_db = list(labels_dbscan).count(-1)

    results.append(
        {
            "Algorithm": "DBSCAN (sklearn)",
            "Time (s)": duration_dbscan,
            "Clusters": n_clus_db,
            "Noise Points": noise_db,
        }
    )
    labels_dict["DBSCAN (sklearn)"] = labels_dbscan

    # --- knndbscan ---
    print(f"Running knndbscan [eps={eps}, min_samples={min_samples}]...")
    start = time.perf_counter()
    labels_knn = knndbscan(X, eps=eps, min_samples=min_samples, n_jobs=-1)
    duration_knn = time.perf_counter() - start

    n_clus_knn = len(set(labels_knn)) - (1 if -1 in labels_knn else 0)
    noise_knn = list(labels_knn).count(-1)

    results.append(
        {
            "Algorithm": "knndbscan",
            "Time (s)": duration_knn,
            "Clusters": n_clus_knn,
            "Noise Points": noise_knn,
        }
    )
    labels_dict["knndbscan"] = labels_knn

    # --- Print summary table ---
    print("\nBenchmark Results:")
    print(f"{'Algorithm':<20} {'Time (s)':>10} {'Clusters':>10} {'Noise Points':>15}")
    print("-" * 60)
    for res in results:
        print(
            f"{res['Algorithm']:<20} {res['Time (s)']:>10.4f} {res['Clusters']:>10} {res['Noise Points']:>15}"
        )

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # KMeans
    axes[0].scatter(X[:, 0], X[:, 1], c=labels_dict["KMeans"], cmap="viridis", s=10)
    axes[0].set_title(f"KMeans (time: {duration_kmeans:.4f}s)")

    # DBSCAN
    axes[1].scatter(
        X[:, 0], X[:, 1], c=labels_dict["DBSCAN (sklearn)"], cmap="viridis", s=10
    )
    axes[1].set_title(f"DBSCAN (time: {duration_dbscan:.4f}s)")

    # knndbscan
    axes[2].scatter(X[:, 0], X[:, 1], c=labels_dict["knndbscan"], cmap="viridis", s=10)
    axes[2].set_title(f"knndbscan (time: {duration_knn:.4f}s)")

    plt.suptitle(f"Clustering Comparison: {dataset_name}")
    plt.tight_layout()

    fig_base_name = f"comparison_{dataset_name.lower().replace(' ', '_')}.png"
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
        cluster_std=0.1,
        random_state=seed,
    )
    run_benchmark(
        X_blobs,
        "Blobs",
        n_clusters=n_blobs,
        eps=0.5,
        min_samples=20,
        seed=seed,
        figs_dir=FIGS_DIR,
    )

    # 2. Moons
    print("Generating Moons...")
    X_moons, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
    run_benchmark(
        X_moons,
        "Moons",
        n_clusters=2,
        eps=0.2,
        min_samples=10,
        seed=seed,
        figs_dir=FIGS_DIR,
    )
