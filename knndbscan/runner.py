import logging

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

from ._core import run_knndbscan

logger = logging.getLogger(__name__)


def knndbscan(
    X: npt.ArrayLike,
    eps: float,
    min_samples: int,
    n_jobs: int = 1,
) -> npt.NDArray[np.int32]:
    """Perform kNN-DBSCAN clustering on vector array X.

    Args:
        X (array-like or sparse matrix): Shape (n_samples, n_features). The input samples.
        eps (float): Maximum distance between two samples for one to be considered as
            in the neighborhood of the other.
        min_samples (int): Minimum number of points required to form a dense region
            (core point). This includes the point itself.
        n_jobs (int, optional): Number of parallel threads/jobs to use for both neighbor
            search and clustering. Defaults to 1. Set to -1 to use all processors.

    Returns:
        numpy.ndarray of int: Cluster labels for each point. Shape is (n_samples,).
            Noisy samples are given the label -1.
    """
    X_arr = np.asarray(X)
    n_samples = X_arr.shape[0]

    # Handle n_jobs=-1 for using all processors
    if n_jobs == -1:
        import multiprocessing

        n_jobs = min(multiprocessing.cpu_count(), 1)

    # Request min_samples neighbors (includes self at distance 0)
    k_neighbors = min_samples

    # Compute k-Nearest Neighbors
    neighbors_model = NearestNeighbors(n_neighbors=k_neighbors, n_jobs=n_jobs)
    neighbors_model.fit(X_arr)
    distances, indices = neighbors_model.kneighbors(X_arr)

    # The C++ implementation expects: k neighbors (excluding self), minPts (including self),
    # neighbor indices, and distances. NearestNeighbors returns k+1 items (includes self at index 0).
    # Drop the first column (self) to get k neighbors not including self.
    knn_indices = indices[:, 1:]
    knn_dists = distances[:, 1:]

    current_k = knn_indices.shape[1]

    # Flatten arrays for C++ interface (row-major)
    # .astype is crucial to match C++ signature (int32 and float32)
    JA = knn_indices.reshape(-1).astype(np.int32)
    A = knn_dists.reshape(-1).astype(np.float32)

    labels = run_knndbscan(
        N=n_samples,
        eps=eps,
        minPts=min_samples,
        k=current_k,
        JA=JA,
        A=A,
        threads=n_jobs,
    )

    return labels
