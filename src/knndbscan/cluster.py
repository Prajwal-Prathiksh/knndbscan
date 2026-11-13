"""Public clustering API: functions for data and for precomputed graphs, and a scikit-learn estimator."""

import os
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin

from ._native import cluster_graph
from .neighbors import Device, knn_graph


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Apply scikit-learn's convention: None is 1, -1 is every CPU this process may use, -2 all but one."""
    if n_jobs is None:
        return 1
    if n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer or None")
    if n_jobs > 0:
        return n_jobs
    # process_cpu_count (3.13+) and sched_getaffinity respect CPU affinity, so a SLURM
    # or cgroup allocation is not oversubscribed; os.cpu_count counts the whole node.
    if hasattr(os, "process_cpu_count"):
        n_cpus = os.process_cpu_count() or 1
    elif hasattr(os, "sched_getaffinity"):
        n_cpus = len(os.sched_getaffinity(0))
    else:
        n_cpus = os.cpu_count() or 1
    return max(1, n_cpus + 1 + n_jobs)


def knn_dbscan_from_graph(
    neighbors: npt.ArrayLike,
    distances: npt.ArrayLike,
    eps: float,
    min_samples: int,
    *,
    n_jobs: int | None = None,
    verbose: bool = False,
) -> npt.NDArray[np.int32]:
    """Cluster a precomputed kNN graph with kNN-DBSCAN.

    Use this with any kNN graph, exact or approximate (for example from FAISS or
    pynndescent). A point is a core point when the distance to its
    ``(min_samples - 1)``-th neighbor is strictly less than ``eps``. Core points are
    linked through their first ``min_samples - 1`` neighbors; further columns only
    serve to attach border points to clusters.

    Args:
        neighbors: Integer neighbor indices, shape ``(n_points, n_neighbors)``. Row ``i``
            lists the neighbors of point ``i`` in ascending order of distance and must not
            contain ``i`` itself. Converted to int32.
        distances: Distances matching ``neighbors``, same shape, finite and non-negative.
            Converted to float32.
        eps: Neighborhood radius; positive.
        min_samples: Number of points, including the point itself, that must lie
            within ``eps`` for it to be a core point. Between 2 and ``n_neighbors + 1``.
        n_jobs: OpenMP threads. ``None`` means 1, ``-1`` every CPU available to the
            process, ``-2`` all but one, and so on.
        verbose: Print the phase timings of the C++ core to the process's standard
            output (not to ``sys.stdout``, so notebooks do not show it).

    Returns:
        Labels of shape ``(n_points,)``, int32: ``0..n_clusters-1`` for points in
        clusters, numbered in order of each cluster's lowest-indexed point, and ``-1``
        for noise. Identical for every ``n_jobs``.

    Raises:
        TypeError: If ``neighbors`` does not have an integer dtype.
        ValueError: If a parameter is out of range or the graph is malformed (bad shape,
            index out of range, self-neighbor, unsorted or invalid distances); the
            message names the first offending row.
        RuntimeError: If the clustering core fails, for example for lack of memory.

    Example:
        >>> from sklearn.neighbors import NearestNeighbors
        >>> distances, neighbors = NearestNeighbors(n_neighbors=9).fit(X).kneighbors()
        >>> labels = knn_dbscan_from_graph(neighbors, distances, eps=0.5, min_samples=10)
    """
    return cluster_graph(
        neighbors, distances, eps, min_samples, _resolve_n_jobs(n_jobs), verbose
    )


def knn_dbscan(
    X: Any,
    eps: float,
    min_samples: int,
    *,
    n_jobs: int | None = None,
    device: Device = "auto",
    verbose: bool = False,
) -> npt.NDArray[np.int32]:
    """Cluster data points with kNN-DBSCAN over their exact Euclidean kNN graph.

    Builds the ``(min_samples - 1)``-nearest-neighbor graph of ``X`` and clusters it
    with :func:`knn_dbscan_from_graph`. Core points are the same as scikit-learn's
    ``DBSCAN(eps, min_samples)`` would find, except that distances equal to ``eps`` do
    not count. Clusters can differ from DBSCAN's: two core points are linked only if one
    is among the other's ``min_samples - 1`` nearest neighbors, so kNN-DBSCAN may split
    a cluster that DBSCAN joins.

    Args:
        X: Data of shape ``(n_samples, n_features)``: a NumPy array, anything
            ``numpy.asarray`` accepts, or a CuPy array.
        eps: Neighborhood radius; positive.
        min_samples: Number of points, including the point itself, that must lie
            within ``eps`` for it to be a core point. Between 2 and ``n_samples``.
        n_jobs: Threads for the neighbor search on the CPU and for clustering. ``None``
            means 1, ``-1`` every CPU available to the process, ``-2`` all but one.
        device: Where to search for neighbors: ``"cpu"`` (scikit-learn), ``"gpu"``
            (cuML; needs the ``gpu`` extra and a CUDA device), or ``"auto"`` (the GPU
            when usable, else the CPU). Clustering always runs on the CPU.
        verbose: Print the phase timings of the C++ core to the process's standard
            output.

    Returns:
        Labels of shape ``(n_samples,)``, int32: ``0..n_clusters-1`` for points in
        clusters, numbered in order of each cluster's lowest-indexed point, and ``-1``
        for noise. Identical for every ``n_jobs``.

    Raises:
        ValueError: If ``min_samples`` is outside ``[2, n_samples]`` or another
            parameter is out of range.
        RuntimeError: If ``device="gpu"`` but no usable GPU is found, or the
            clustering core fails.

    Example:
        >>> from sklearn.datasets import make_blobs
        >>> X, _ = make_blobs(n_samples=1000, centers=3, random_state=0)
        >>> labels = knn_dbscan(X, eps=0.5, min_samples=10, device="cpu")
    """
    model = KNNDBSCAN(eps, min_samples, n_jobs=n_jobs, device=device, verbose=verbose)
    return model.fit(X).labels_


class KNNDBSCAN(ClusterMixin, BaseEstimator):  # type: ignore[misc]
    """kNN-DBSCAN clustering as a scikit-learn estimator.

    The estimator form of :func:`knn_dbscan`, usable in scikit-learn pipelines and model
    selection. See :func:`knn_dbscan` for how the results relate to DBSCAN.

    Args:
        eps: Neighborhood radius; positive.
        min_samples: Number of points, including the point itself, that must lie
            within ``eps`` for it to be a core point. At least 2.
        n_jobs: Threads for the neighbor search and clustering; ``None`` means 1,
            ``-1`` every CPU available to the process.
        device: ``"cpu"``, ``"gpu"`` or ``"auto"``; where to search for neighbors.
        verbose: Print the phase timings of the C++ core to standard output.

    Attributes:
        labels_: Labels of the fitted data, int32: ``0..n_clusters-1``, ``-1`` for noise.
        core_sample_indices_: Indices of the core points, in increasing order.

    Example:
        >>> labels = KNNDBSCAN(eps=0.5, min_samples=10, device="cpu").fit_predict(X)
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        *,
        n_jobs: int | None = None,
        device: Device = "auto",
        verbose: bool = False,
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.device = device
        self.verbose = verbose

    def fit(self, X: Any, y: None = None) -> "KNNDBSCAN":
        """Cluster ``X`` and store ``labels_`` and ``core_sample_indices_``.

        Args:
            X: Data of shape ``(n_samples, n_features)``; see :func:`knn_dbscan`.
            y: Ignored; present for scikit-learn API compatibility.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If ``min_samples`` is outside ``[2, n_samples]`` or another
                parameter is out of range.
            RuntimeError: If ``device="gpu"`` but no usable GPU is found, or the
                clustering core fails.
        """
        n_samples = len(X)
        if not 2 <= self.min_samples <= n_samples:
            raise ValueError(
                f"min_samples must be in [2, n_samples], got {self.min_samples} "
                f"with n_samples = {n_samples}"
            )
        n_jobs = _resolve_n_jobs(self.n_jobs)
        neighbors, distances = knn_graph(
            X, self.min_samples - 1, device=self.device, n_jobs=n_jobs
        )
        self.labels_ = cluster_graph(
            neighbors, distances, self.eps, self.min_samples, n_jobs, self.verbose
        )
        # The same float32 comparison as the C++ core, so the two agree at eps exactly.
        is_core = distances[:, self.min_samples - 2] < np.float32(self.eps)
        self.core_sample_indices_ = np.flatnonzero(is_core)
        return self
