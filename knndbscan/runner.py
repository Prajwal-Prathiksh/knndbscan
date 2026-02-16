import logging
import time
from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

try:
    import cupy as cp  # type: ignore
    from cuml.neighbors import NearestNeighbors as CuMLNearestNeighbors  # type: ignore

    HAS_GPU = True
except ImportError:
    CuMLNearestNeighbors = None
    cp = None
    HAS_GPU = False

from ._core import run_knndbscan

logger = logging.getLogger(__name__)


def knn_dbscan(
    X: npt.ArrayLike,
    eps: float,
    min_samples: int,
    n_jobs: int = 1,
    verbose: bool = False,
    device: Literal["auto", "cpu", "gpu"] = "auto",
    *,
    k_neighbors: int | None = None,
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
        verbose (bool, optional): Whether to print timing and debug information from the
            core C++ implementation. Defaults to False.
        device (str, optional): Device to use for neighbor search.
            "auto": Use GPU if available, else CPU.
            "cpu": Force CPU usage.
            "gpu": Force GPU usage. Raises ImportError if GPU libraries are missing.
        k_neighbors (int, optional): Number of nearest neighbors to build the
            neighborhood graph. If None, defaults to min_samples.

    Returns:
        numpy.ndarray of int: Cluster labels for each point. Shape is (n_samples,).
            Noisy samples are given the label -1.

    Note:
        The library automatically configures OpenMPI environment variables for optimal
        single-node performance. To disable this and use custom MPI settings, set:
        ``export KNNDBSCAN_AUTO_CONFIGURE_MPI=0`` before importing knndbscan.
    """
    if HAS_GPU and isinstance(X, (cp.ndarray,)):  # type: ignore
        X_arr = X
        input_is_cupy = True
    else:
        X_arr = np.asarray(X)
        input_is_cupy = False

    n_samples = X_arr.shape[0]  # type: ignore

    # Handle n_jobs=-1 for using all processors (only relevant for CPU)
    if n_jobs == -1:
        import multiprocessing

        n_jobs = multiprocessing.cpu_count()

    # Determine device usage
    use_gpu = False
    if device == "gpu":
        if not HAS_GPU:
            raise ImportError(
                "GPU support requested but 'cuml' or 'cupy' not installed. "
                "Install with 'pip install knndbscan[gpu]'."
            )
        use_gpu = True
    elif device == "auto":
        # Check if GPU libraries are available
        if HAS_GPU:
            use_gpu = True

    # Request min_samples neighbors (includes self at distance 0)
    k_neighbors = min_samples if k_neighbors is None else k_neighbors

    t0 = time.perf_counter()

    # Initialize variables for C++ side (needs to be CPU numpy)
    distances_cpu = None
    indices_cpu = None

    if use_gpu and cp is not None and CuMLNearestNeighbors is not None:
        # Ensure input is on GPU
        if not input_is_cupy:
            # Transfer to GPU if it wasn't already
            X_gpu = cp.asarray(X_arr, dtype=np.float32)
        else:
            # Already on GPU, just ensure type
            X_gpu = X_arr.astype(np.float32, copy=False)  # type: ignore

        # cuML NearestNeighbors
        logger.debug("Starting k-NN search on GPU using cuML")
        neighbors_model = CuMLNearestNeighbors(n_neighbors=k_neighbors)
        neighbors_model.fit(X_gpu)
        distances_gpu, indices_gpu = neighbors_model.kneighbors(X_gpu)

        # Bring back to CPU as numpy arrays for C++ binding
        distances_cpu = cp.asnumpy(distances_gpu)
        indices_cpu = cp.asnumpy(indices_gpu)

    else:
        # Fallback ensuring X_arr is numpy
        if input_is_cupy:
            X_cpu = cp.asnumpy(X_arr)  # type: ignore
        else:
            X_cpu = X_arr

        # Compute k-Nearest Neighbors
        logger.debug(f"Starting k-NN search on CPU with n_jobs={n_jobs:,}")
        neighbors_model = NearestNeighbors(n_neighbors=k_neighbors, n_jobs=n_jobs)
        neighbors_model.fit(X_cpu)  # type: ignore
        distances_cpu, indices_cpu = neighbors_model.kneighbors(X_cpu)  # type: ignore

    logger.debug(f"k-NN search completed in {time.perf_counter() - t0:.4f} seconds")

    # The C++ implementation expects: k neighbors (excluding self), minPts (including self),
    # neighbor indices, and distances. NearestNeighbors returns k+1 items (includes self at index 0).
    # Drop the first column (self) to get k neighbors not including self.
    knn_indices = indices_cpu[:, 1:]
    knn_dists = distances_cpu[:, 1:]

    current_k = knn_indices.shape[1]

    # Flatten arrays for C++ interface (row-major)
    # .astype is crucial to match C++ signature (int32 and float32)
    JA = knn_indices.reshape(-1).astype(np.int32)
    A = knn_dists.reshape(-1).astype(np.float32)

    logger.debug("Starting DBSCAN clustering (C++ core)")
    t0_dbscan = time.perf_counter()

    labels = run_knndbscan(
        N=n_samples,
        eps=eps,
        minPts=min_samples,
        k=current_k,
        JA=JA,
        A=A,
        mpi_threads=n_jobs,
        verbose=verbose,
    )
    logger.debug(
        f"DBSCAN clustering completed in {time.perf_counter() - t0_dbscan:.4f} seconds"
    )

    # Return same type as input
    if input_is_cupy and cp is not None:
        return cp.asarray(labels)  # type: ignore

    return labels
