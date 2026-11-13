"""ctypes binding to the knndbscan shared library; the C contract is in cpp/include/knndbscan.h.

The library links neither Python nor NumPy, which is what lets one wheel per platform
serve every Python 3 version. ctypes releases the GIL for the duration of each call.
"""

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

_LIBRARY_FILES = {"darwin": "libknndbscan.dylib", "win32": "knndbscan.dll"}
_ERROR_INVALID_ARGUMENT = 1
_INT32 = np.iinfo(np.int32)


def _load_library() -> ctypes.CDLL:
    """Load $KNNDBSCAN_LIBRARY if set (e.g. a sanitizer build), else the library on __path__.

    A wheel installs the library next to this file; an editable install keeps the Python
    sources in src/ and the built library in site-packages, and lists both on __path__.
    """
    override = os.environ.get("KNNDBSCAN_LIBRARY")
    if override:
        path = Path(override)
    else:
        name = _LIBRARY_FILES.get(sys.platform, "libknndbscan.so")
        candidates = [Path(entry) / name for entry in sys.modules[__package__].__path__]
        found = next(
            (candidate for candidate in candidates if candidate.exists()), None
        )
        if found is None:
            raise ImportError(
                f"{name} not found in {[str(c.parent) for c in candidates]}"
            )
        path = found
    library = ctypes.CDLL(str(path))
    library.knndbscan_cluster.argtypes = [
        np.ctypeslib.ndpointer(np.int32, ndim=2, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float32, ndim=2, flags="C_CONTIGUOUS"),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        np.ctypeslib.ndpointer(np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ]
    library.knndbscan_cluster.restype = ctypes.c_int
    library.knndbscan_last_error.argtypes = []
    library.knndbscan_last_error.restype = ctypes.c_char_p
    return library


_library = _load_library()


def _as_int32_indices(neighbors: npt.ArrayLike) -> npt.NDArray[np.int32]:
    """Convert to C-contiguous int32, refusing values a cast would wrap into valid indices."""
    array = np.asarray(neighbors)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"neighbors must have an integer dtype, got {array.dtype}")
    if (
        array.dtype != np.int32
        and array.size
        and (array.min() < _INT32.min or array.max() > _INT32.max)
    ):
        raise ValueError("neighbor index outside [0, n_points)")
    return np.ascontiguousarray(array, dtype=np.int32)


def cluster_graph(
    neighbors: npt.ArrayLike,
    distances: npt.ArrayLike,
    eps: float,
    min_samples: int,
    n_threads: int,
    verbose: bool,
) -> npt.NDArray[np.int32]:
    """Convert the graph to the C layout and run knndbscan_cluster on it.

    The library checks the graph's contents; this function checks only what must hold
    before the call: dtypes, matching 2-D shapes, and a size that fits the int32
    arguments (ctypes would silently wrap a larger one). Raises ValueError for
    arguments the C contract rejects and RuntimeError for failures inside the
    algorithm, each with the library's message.
    """
    neighbors = _as_int32_indices(neighbors)
    distances = np.ascontiguousarray(distances, dtype=np.float32)
    if neighbors.ndim != 2 or neighbors.shape != distances.shape:
        raise ValueError(
            "neighbors and distances must be 2-D arrays of the same shape "
            f"(n_points, n_neighbors), got {neighbors.shape} and {distances.shape}"
        )
    if neighbors.size > _INT32.max:
        raise ValueError("n_points * n_neighbors must be below 2^31")
    labels = np.empty(neighbors.shape[0], dtype=np.int32)
    status = _library.knndbscan_cluster(
        neighbors,
        distances,
        *neighbors.shape,
        eps,
        min_samples,
        n_threads,
        int(verbose),
        labels,
    )
    if status != 0:
        message = _library.knndbscan_last_error().decode()
        if status == _ERROR_INVALID_ARGUMENT:
            raise ValueError(message)
        raise RuntimeError(message)
    return labels
