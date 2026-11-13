"""Exact kNN graphs for kNN-DBSCAN, on the CPU with scikit-learn or on the GPU with cuML.

Graphs follow the package convention: rows sorted by distance, the point itself excluded,
returned as host NumPy arrays because the clustering library runs on the CPU.
"""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

Device = Literal["auto", "cpu", "gpu"]


def gpu_available() -> bool:
    """True when CuPy and cuML import and CUDA reports at least one device."""
    try:
        import cuml  # noqa: F401
        import cupy
    except ImportError:
        return False
    try:
        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except cupy.cuda.runtime.CUDARuntimeError:
        return False


def resolve_device(device: Device) -> Literal["cpu", "gpu"]:
    """Map "auto" to the GPU when one is usable; raise if "gpu" is requested but unusable."""
    if device == "cpu":
        return "cpu"
    if device not in ("auto", "gpu"):
        raise ValueError(f"device must be 'auto', 'cpu' or 'gpu', got {device!r}")
    if gpu_available():
        return "gpu"
    if device == "gpu":
        raise RuntimeError(
            "device='gpu' needs CuPy, cuML and a CUDA device; "
            "install them with `pip install knndbscan[gpu]`"
        )
    return "cpu"


def _is_cupy_array(X: Any) -> bool:
    return type(X).__module__.split(".")[0] == "cupy"


def _drop_self(
    indices: npt.NDArray[np.integer], distances: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
    """Remove each point from its own row of a k+1-neighbor query.

    With duplicate points the point need not be in column 0, and with more than k
    duplicates it may be missing; such a row drops its last column instead, which is a
    duplicate at distance 0 like the rest.
    """
    is_self = indices == np.arange(len(indices))[:, None]
    is_self[~is_self.any(axis=1), -1] = True
    keep = ~is_self
    n_points = len(indices)
    return indices[keep].reshape(n_points, -1), distances[keep].reshape(n_points, -1)


def knn_graph(
    X: Any, n_neighbors: int, *, device: Device, n_jobs: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Return (neighbors, distances) of shape (n_points, n_neighbors) under the Euclidean metric.

    `n_jobs` applies to the CPU search only. A CuPy input stays on the GPU for a GPU
    search and is copied to the host for a CPU one.
    """
    if resolve_device(device) == "gpu":
        import cupy
        from cuml.neighbors import NearestNeighbors as CumlNearestNeighbors

        X_gpu = cupy.asarray(X, dtype=cupy.float32)
        model = CumlNearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_gpu)
        gpu_distances, gpu_indices = model.kneighbors(X_gpu)
        indices, distances = _drop_self(
            cupy.asnumpy(gpu_indices), cupy.asnumpy(gpu_distances)
        )
    else:
        if _is_cupy_array(X):
            X = X.get()
        # kneighbors() without X excludes each query point by index, so duplicates of
        # a point stay in its row.
        model = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs).fit(X)
        distances, indices = model.kneighbors()
    return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=np.float32)
