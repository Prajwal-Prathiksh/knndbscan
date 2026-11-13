"""GPU neighbor search must give the CPU result; skipped without CuPy, cuML and a CUDA device."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from knndbscan import KNNDBSCAN, knn_dbscan
from knndbscan.neighbors import gpu_available, resolve_device

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not gpu_available(), reason="no usable GPU"),
]


@pytest.fixture(scope="module")
def blobs() -> np.ndarray:
    # Well separated, so no distance lies within float32 rounding of eps.
    return make_blobs(n_samples=3000, centers=4, cluster_std=0.5, random_state=0)[0]


def test_auto_selects_the_gpu() -> None:
    assert resolve_device("auto") == "gpu"


def test_gpu_and_cpu_searches_give_the_same_clustering(blobs: np.ndarray) -> None:
    cpu = KNNDBSCAN(0.5, 10, device="cpu").fit(blobs)
    gpu = KNNDBSCAN(0.5, 10, device="gpu").fit(blobs)
    np.testing.assert_array_equal(gpu.labels_, cpu.labels_)
    np.testing.assert_array_equal(gpu.core_sample_indices_, cpu.core_sample_indices_)


def test_cupy_input_is_accepted_on_both_devices(blobs: np.ndarray) -> None:
    import cupy

    expected = knn_dbscan(blobs, 0.5, 10, device="cpu")
    for device in ("cpu", "gpu"):
        labels = knn_dbscan(cupy.asarray(blobs), 0.5, 10, device=device)  # type: ignore[arg-type]
        np.testing.assert_array_equal(labels, expected)
