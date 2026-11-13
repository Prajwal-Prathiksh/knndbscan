"""Tests of the public API on the CPU: agreement with DBSCAN, invariances and input checks."""

from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

import knndbscan
from knndbscan import KNNDBSCAN, knn_dbscan, knn_dbscan_from_graph
from knndbscan.cluster import _resolve_n_jobs
from knndbscan.neighbors import _drop_self, knn_graph, resolve_device

EPS = 0.5
MIN_SAMPLES = 10


@pytest.fixture(scope="module")
def blobs() -> np.ndarray:
    X, _ = make_blobs(n_samples=2000, centers=4, cluster_std=0.6, random_state=0)
    return X


@pytest.fixture(scope="module")
def graph(blobs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances, neighbors = NearestNeighbors(n_neighbors=30).fit(blobs).kneighbors()
    return neighbors, distances


def test_version_matches_metadata() -> None:
    assert knndbscan.__version__ == version("knndbscan")


def test_core_points_clusters_and_noise_match_dbscan_on_blobs(
    blobs: np.ndarray,
) -> None:
    reference = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(blobs)
    model = KNNDBSCAN(EPS, MIN_SAMPLES, device="cpu").fit(blobs)
    core = reference.core_sample_indices_
    np.testing.assert_array_equal(model.core_sample_indices_, core)
    assert adjusted_rand_score(reference.labels_[core], model.labels_[core]) == 1.0
    # A border point within eps of two clusters may join either one, so only the
    # noise set is compared outside the core points.
    np.testing.assert_array_equal(model.labels_ == -1, reference.labels_ == -1)


def test_labels_are_contiguous_from_zero_with_minus_one_for_noise(
    blobs: np.ndarray,
) -> None:
    labels = knn_dbscan(blobs, EPS, MIN_SAMPLES, device="cpu")
    assert labels.dtype == np.int32
    clusters = np.unique(labels[labels >= 0])
    np.testing.assert_array_equal(clusters, np.arange(len(clusters)))
    assert labels.min() == -1


@pytest.mark.parametrize(("eps", "min_samples"), [(0.3, 5), (0.5, 10)])
def test_labels_do_not_depend_on_thread_count_or_run(
    eps: float, min_samples: int
) -> None:
    # The core's own numbering varies with thread scheduling on these parameters.
    X, _ = make_blobs(n_samples=5000, centers=6, cluster_std=0.7, random_state=0)
    neighbors, distances = knn_graph(X, min_samples - 1, device="cpu", n_jobs=1)
    reference = knn_dbscan_from_graph(neighbors, distances, eps, min_samples, n_jobs=1)
    for n_jobs in (2, 3, 4, 8, 4, 4, 4):
        labels = knn_dbscan_from_graph(
            neighbors, distances, eps, min_samples, n_jobs=n_jobs
        )
        np.testing.assert_array_equal(labels, reference)


def test_function_estimator_and_graph_api_agree(blobs: np.ndarray) -> None:
    from_data = knn_dbscan(blobs, EPS, MIN_SAMPLES, device="cpu")
    from_estimator = KNNDBSCAN(EPS, MIN_SAMPLES, device="cpu").fit_predict(blobs)
    neighbors, distances = knn_graph(blobs, MIN_SAMPLES - 1, device="cpu", n_jobs=1)
    from_graph = knn_dbscan_from_graph(neighbors, distances, EPS, MIN_SAMPLES)
    np.testing.assert_array_equal(from_data, from_estimator)
    np.testing.assert_array_equal(from_data, from_graph)


def test_estimator_clones_with_its_parameters() -> None:
    model = KNNDBSCAN(0.3, 7, n_jobs=2, device="cpu")
    assert clone(model).get_params() == model.get_params()


def test_input_layout_and_dtype_do_not_change_labels(
    graph: tuple[np.ndarray, np.ndarray],
) -> None:
    neighbors, distances = graph
    reference = knn_dbscan_from_graph(neighbors, distances, EPS, MIN_SAMPLES)
    strided = (
        np.repeat(neighbors, 2, axis=1)[:, ::2],
        np.repeat(distances, 2, axis=1)[:, ::2],
    )
    fortran = (np.asfortranarray(neighbors), np.asfortranarray(distances))
    narrow_types = (neighbors.astype(np.int32), distances.astype(np.float32))
    for variant in (strided, fortran, narrow_types):
        np.testing.assert_array_equal(
            knn_dbscan_from_graph(*variant, EPS, MIN_SAMPLES), reference
        )


def test_concurrent_calls_return_the_serial_results(blobs: np.ndarray) -> None:
    datasets = [blobs[i::4] for i in range(4)]
    expected = [knn_dbscan(X, EPS, 5, device="cpu") for X in datasets]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda X: knn_dbscan(X, EPS, 5, device="cpu"), datasets * 4)
        )
    for i, labels in enumerate(results):
        np.testing.assert_array_equal(labels, expected[i % 4])


def test_duplicate_points_are_clustered_together() -> None:
    X = np.repeat(make_blobs(n_samples=300, centers=3, random_state=1)[0], 3, axis=0)
    neighbors, _ = knn_graph(X, 8, device="cpu", n_jobs=1)
    assert not (neighbors == np.arange(len(X))[:, None]).any()
    labels = knn_dbscan(X, 0.5, 9, device="cpu")
    assert (labels[0::3] == labels[1::3]).all() and (labels[0::3] == labels[2::3]).all()


def test_drop_self_handles_points_outside_column_zero_and_missing() -> None:
    indices = np.array([[1, 0, 2], [1, 0, 2], [0, 1, 3], [3, 2, 1]])
    distances = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 2, 3]], dtype=float)
    kept, kept_distances = _drop_self(indices, distances)
    np.testing.assert_array_equal(kept, [[1, 2], [0, 2], [0, 1], [2, 1]])
    np.testing.assert_array_equal(kept_distances, [[0, 1], [0, 1], [0, 0], [2, 3]])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"min_samples": 1}, "min_samples must be in"),
        ({"min_samples": 32}, "min_samples must be in"),
        ({"eps": 0.0}, "eps must be positive"),
        ({"eps": -1.0}, "eps must be positive"),
        ({"eps": float("inf")}, "eps must be positive"),
        ({"n_jobs": 0}, "n_jobs"),
    ],
)
def test_invalid_parameters_raise(
    graph: tuple[np.ndarray, np.ndarray], change: dict, message: str
) -> None:
    neighbors, distances = graph
    arguments = {"eps": EPS, "min_samples": MIN_SAMPLES} | change
    with pytest.raises(ValueError, match=message):
        knn_dbscan_from_graph(neighbors, distances, **arguments)


def _corrupt_row_5(neighbors: np.ndarray, distances: np.ndarray, kind: str) -> None:
    if kind == "index_too_large":
        neighbors[5, 3] = len(neighbors)
    elif kind == "index_negative":
        neighbors[5, 3] = -1
    elif kind == "self_neighbor":
        neighbors[5, 3] = 5
    elif kind == "unsorted":
        distances[5, 3] = distances[5, 4] + 1
    elif kind == "nan":
        distances[5, 3] = np.nan
    elif kind == "negative_distance":
        distances[5, 0] = -1


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("index_too_large", "outside"),
        ("index_negative", "outside"),
        ("self_neighbor", "own neighbor"),
        ("unsorted", "ascending"),
        ("nan", "NaN"),
        ("negative_distance", "negative"),
    ],
)
def test_malformed_graphs_raise_naming_the_first_bad_row(
    graph: tuple[np.ndarray, np.ndarray], kind: str, message: str
) -> None:
    neighbors, distances = graph[0].copy(), graph[1].copy()
    _corrupt_row_5(neighbors, distances, kind)
    with pytest.raises(ValueError, match=f"row 5: .*{message}"):
        knn_dbscan_from_graph(neighbors, distances, EPS, MIN_SAMPLES)


def test_malformed_shapes_and_dtypes_raise(
    graph: tuple[np.ndarray, np.ndarray],
) -> None:
    neighbors, distances = graph
    with pytest.raises(ValueError, match="same shape"):
        knn_dbscan_from_graph(neighbors, distances[:, :5], EPS, MIN_SAMPLES)
    with pytest.raises(ValueError, match="2-D"):
        knn_dbscan_from_graph(neighbors.ravel(), distances.ravel(), EPS, MIN_SAMPLES)
    with pytest.raises(TypeError, match="integer dtype"):
        knn_dbscan_from_graph(neighbors.astype(float), distances, EPS, MIN_SAMPLES)
    wrapping = neighbors.astype(np.int64)
    wrapping[0, 0] = 2**32  # would become 0 under an int32 cast
    with pytest.raises(ValueError, match="outside"):
        knn_dbscan_from_graph(wrapping, distances, EPS, MIN_SAMPLES)


def test_min_samples_must_fit_the_data() -> None:
    with pytest.raises(ValueError, match="min_samples must be in"):
        knn_dbscan(np.zeros((5, 2)), EPS, 6, device="cpu")


def test_device_resolution() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "gpu")
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("tpu")  # type: ignore[arg-type]


def test_negative_n_jobs_counts_back_from_the_available_cpus() -> None:
    all_cpus = _resolve_n_jobs(-1)
    assert all_cpus >= 1
    assert _resolve_n_jobs(-2) == max(1, all_cpus - 1)
    assert _resolve_n_jobs(-10_000) == 1
    assert _resolve_n_jobs(None) == 1
