"""The library must reproduce kNN-DBSCAN as defined in the paper, on any valid kNN graph.

Random directed graphs exercise what real data rarely does: cycles in Boruvka's
pointer jumping, ties, tiny components and eps equal to a distance.
"""

import logging

import numpy as np
import pytest
from reference import (
    assert_matches_reference,
    eps_at_quantile,
    random_graph,
    reference_clustering,
)
from sklearn.datasets import make_blobs, make_moons
from sklearn.neighbors import NearestNeighbors

from knndbscan import knn_dbscan, knn_dbscan_from_graph


def _check(neighbors, distances, eps, min_samples, n_jobs=1) -> np.ndarray:
    labels = knn_dbscan_from_graph(
        neighbors, distances, eps, min_samples, n_jobs=n_jobs
    )
    reference = reference_clustering(neighbors, distances, eps, min_samples)
    assert_matches_reference(labels, reference)
    return labels


@pytest.mark.parametrize("seed", range(24))
def test_random_directed_graphs_match_the_definitions(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_points = int(rng.choice([20, 150, 1200]))
    n_neighbors = int(rng.integers(2, 12))
    min_samples = int(rng.integers(2, n_neighbors + 2))
    neighbors, distances = random_graph(rng, n_points, n_neighbors)
    eps = eps_at_quantile(distances, min_samples, rng.uniform(0.2, 0.9))
    _check(neighbors, distances, eps, min_samples, n_jobs=int(rng.choice([1, 4])))


@pytest.mark.parametrize("levels", [1, 2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_tied_distances_match_the_definitions(levels: int, seed: int) -> None:
    # levels=1 makes every distance 0, so every edge ties with every other; otherwise eps
    # is the lowest nonzero level, which many distances equal exactly.
    rng = np.random.default_rng(100 + seed)
    neighbors, distances = random_graph(rng, 400, 6, levels=levels)
    eps = 0.5 if levels == 1 else float(np.float32(1 / levels))
    _check(neighbors, distances, eps, 5, n_jobs=4)


@pytest.mark.parametrize(
    ("make_data", "eps", "min_samples"),
    [
        (
            lambda: make_blobs(3000, centers=5, cluster_std=0.7, random_state=0)[0],
            0.4,
            8,
        ),
        (lambda: make_moons(3000, noise=0.07, random_state=0)[0], 0.06, 12),
        (lambda: np.random.default_rng(0).normal(size=(2000, 16)), 4.0, 10),
    ],
)
def test_exact_knn_graphs_of_data_match_the_definitions(
    make_data, eps, min_samples
) -> None:
    distances, neighbors = (
        NearestNeighbors(n_neighbors=20).fit(make_data()).kneighbors()
    )
    _check(neighbors, distances, eps, min_samples, n_jobs=3)


def test_directed_cycle_of_figure_4_is_one_cluster() -> None:
    # Fig. 4 of the paper: every point's nearest neighbor is the next one around the
    # cycle V1 -> V2 -> V3 -> V4 -> V1, so pointer jumping alone never terminates.
    neighbors = np.array([[1, 3], [2, 3], [3, 1], [0, 2]])
    distances = np.array([[0.12, 0.13], [0.15, 0.17], [0.14, 0.15], [0.13, 0.14]])
    labels = _check(neighbors, distances, eps=1.0, min_samples=3)
    np.testing.assert_array_equal(labels, [0, 0, 0, 0])


def test_core_border_and_noise_points_of_a_hand_checked_example() -> None:
    # eps = 1, min_samples = 3: 0.0..1.5 each have two neighbors within 1 (core); 2.4 has
    # one, the core point 1.5 (border); 10.0 and 10.5 have only each other (noise).
    X = np.array([0.0, 0.5, 1.0, 1.5, 2.4, 10.0, 10.5])[:, None]
    labels = knn_dbscan(X, eps=1.0, min_samples=3, device="cpu")
    np.testing.assert_array_equal(labels, [0, 0, 0, 0, 0, -1, -1])


def test_degenerate_parameters() -> None:
    X, _ = make_blobs(300, centers=3, random_state=2)
    distances, neighbors = NearestNeighbors(n_neighbors=5).fit(X).kneighbors()
    assert (_check(neighbors, distances, 1e-9, 5) == -1).all()
    assert (_check(neighbors, distances, 1e9, 2) >= 0).all()
    pair = np.array([[1], [0]]), np.array([[0.5], [0.5]])
    np.testing.assert_array_equal(_check(*pair, 0.5, 2), [0, 0])
    np.testing.assert_array_equal(_check(*pair, 0.4, 2), [-1, -1])


def test_components_smaller_than_min_samples_are_clusters_of_their_own() -> None:
    # Separated pairs: with min_samples=2 each pair is a cluster, though far below
    # any size a cluster of DBSCAN's "minimum points" might suggest.
    X = (
        np.repeat(np.arange(0.0, 100.0, 10.0), 2)[:, None]
        + np.tile([0.0, 0.1], 10)[:, None]
    )
    labels = knn_dbscan(X, eps=0.2, min_samples=2, device="cpu")
    np.testing.assert_array_equal(labels, np.repeat(np.arange(10), 2))


def test_core_logs_through_python_logging_and_never_prints(
    capfd: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    # Random directed graphs make the core break cycles, so every log site runs. Nothing
    # may reach the file descriptors (where C++ would write, beneath sys.stdout), and
    # records appear only at the levels the "knndbscan" logger enables.
    rng = np.random.default_rng(10_003)
    neighbors, distances = random_graph(rng, 400, 6)
    eps = eps_at_quantile(distances, 5, 0.8)

    knn_dbscan_from_graph(neighbors, distances, eps, 5, n_jobs=8)
    assert not [r for r in caplog.records if r.name.startswith("knndbscan")]

    with caplog.at_level(logging.INFO, logger="knndbscan"):
        knn_dbscan_from_graph(neighbors, distances, eps, 5, n_jobs=8)
    info = [r.getMessage() for r in caplog.records if r.name == "knndbscan.core"]
    assert any(m.startswith("core points:") for m in info)
    assert any(m.startswith("cut MST:") for m in info)
    assert all(r.levelno == logging.INFO for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="knndbscan"):
        knn_dbscan_from_graph(neighbors, distances, eps, 5, n_jobs=8)
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(m.startswith("local MST step") for m in debug)

    out, err = capfd.readouterr()
    assert out == "" and err == ""


def test_neighbor_search_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    X, _ = make_blobs(200, centers=2, random_state=0)
    with caplog.at_level(logging.INFO, logger="knndbscan"):
        knn_dbscan(X, 0.5, 5, device="cpu")
    assert any(
        r.name == "knndbscan.neighbors" and "on the CPU" in r.getMessage()
        for r in caplog.records
    )
