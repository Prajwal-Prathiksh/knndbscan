"""The paper's results about kNN-DBSCAN, and the invariances its definition implies.

Lemma 3.1 and Theorem 3.4 relate the output to scikit-learn's DBSCAN; the remaining
tests transform the input in ways that must leave the clustering unchanged (or change
it predictably), which catches numerical bugs no fixed expected output would.
"""

import numpy as np
import pytest
from reference import eps_at_quantile, random_graph, same_partition
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.neighbors import NearestNeighbors

from knndbscan import KNNDBSCAN, knn_dbscan_from_graph

MIN_SAMPLES = 10


def _grid() -> np.ndarray:
    # Many neighbor distances equal 1.0 exactly, the eps used with it: ties at eps.
    return np.stack(np.meshgrid(np.arange(12.0), np.arange(12.0)), -1).reshape(-1, 2)


def _anisotropic() -> np.ndarray:
    X, _ = make_blobs(3000, centers=4, cluster_std=0.5, random_state=3)
    return X @ np.array([[0.6, -0.6], [-0.4, 0.8]])


DATASETS = {
    # name: (data, [(eps, min_samples), ...])
    "blobs": (make_blobs(3000, centers=6, cluster_std=0.7, random_state=0)[0], [(0.3, 5), (0.5, 10), (1.0, 30)]),
    "moons": (make_moons(3000, noise=0.08, random_state=0)[0], [(0.05, 10), (0.1, 20), (0.5, 10)]),
    "high-dimensional": (np.random.default_rng(1).normal(size=(2000, 20)) + np.repeat(np.eye(20)[:4] * 6, 500, axis=0), [(5.0, 10), (6.0, 25)]),
    "anisotropic": (_anisotropic(), [(0.2, 8), (0.35, 15)]),
    "grid": (_grid(), [(1.0, 5), (1.0, 4), (1.5, 9)]),
}  # fmt: skip
CASES = [
    (name, eps, ms) for name, (_, params) in DATASETS.items() for eps, ms in params
]


@pytest.fixture(
    scope="module", params=CASES, ids=lambda case: f"{case[0]}-eps{case[1]}-ms{case[2]}"
)
def fitted(request):
    name, eps, min_samples = request.param
    X = DATASETS[name][0]
    reference = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    model = KNNDBSCAN(eps, min_samples, device="cpu").fit(X)
    return reference, model


def test_lemma_3_1_core_border_and_noise_points_are_dbscans(fitted) -> None:
    reference, model = fitted
    np.testing.assert_array_equal(
        model.core_sample_indices_, reference.core_sample_indices_
    )
    np.testing.assert_array_equal(model.labels_ == -1, reference.labels_ == -1)


def test_theorem_3_4_clusters_refine_dbscans_and_coincide_when_counts_match(
    fitted,
) -> None:
    reference, model = fitted
    core = reference.core_sample_indices_
    ours, theirs = model.labels_[core], reference.labels_[core]
    # (i) every kNN-DBSCAN cluster lies inside one DBSCAN cluster
    for cluster in np.unique(ours):
        assert len(np.unique(theirs[ours == cluster])) == 1
    n_ours, n_theirs = len(np.unique(ours)), len(np.unique(theirs))
    assert n_ours >= n_theirs
    # (ii) with as many clusters, the clusterings coincide
    if n_ours == n_theirs:
        assert same_partition(ours, theirs)


@pytest.mark.parametrize("seed", range(4))
def test_larger_eps_only_adds_core_points_and_merges_clusters(seed: int) -> None:
    rng = np.random.default_rng(seed)
    neighbors, distances = random_graph(rng, 800, 8)
    eps_values = sorted(eps_at_quantile(distances, 6, q) for q in (0.2, 0.4, 0.6, 0.8))
    labels = [knn_dbscan_from_graph(neighbors, distances, eps, 6) for eps in eps_values]
    cores = [distances[:, 4] <= np.float32(eps) for eps in eps_values]
    for (small, core_small), (large, core_large) in zip(
        zip(labels, cores, strict=True),
        zip(labels[1:], cores[1:], strict=True),
        strict=False,
    ):
        assert not (core_small & ~core_large).any()
        for cluster in np.unique(small[core_small]):
            assert len(np.unique(large[core_small & (small == cluster)])) == 1


@pytest.mark.parametrize("exponent", [-100, -20, 13, 14, 16, 24, 100])
def test_scaling_distances_and_eps_by_a_power_of_two_changes_nothing(
    exponent: int,
) -> None:
    # Multiplying by 2^k is exact in floating point, so any change is a bug; large
    # factors catch weights compared against a finite "no edge" sentinel.
    X, _ = make_blobs(2000, centers=4, cluster_std=0.6, random_state=0)
    distances, neighbors = NearestNeighbors(n_neighbors=12).fit(X).kneighbors()
    distances = distances.astype(np.float32)
    scale = np.float32(2.0**exponent)
    expected = knn_dbscan_from_graph(neighbors, distances, 0.5, MIN_SAMPLES)
    scaled = knn_dbscan_from_graph(
        neighbors, distances * scale, 0.5 * float(scale), MIN_SAMPLES
    )
    np.testing.assert_array_equal(scaled, expected)


@pytest.mark.parametrize(
    "transform", [lambda d: 3 * d + 7, lambda d: d * d], ids=["affine", "square"]
)
def test_strictly_increasing_transforms_of_distances_and_eps_change_nothing(
    transform,
) -> None:
    # The algorithm only compares distances, so any order-preserving map of the
    # distances and eps must give identical labels. Integer distances keep both maps
    # exact in float32.
    rng = np.random.default_rng(7)
    neighbors, _ = random_graph(rng, 1000, 8)
    distances = np.sort(rng.integers(1, 2000, size=neighbors.shape), axis=1).astype(
        np.float32
    )
    eps = 900.5
    expected = knn_dbscan_from_graph(neighbors, distances, eps, 6)
    transformed = knn_dbscan_from_graph(
        neighbors, transform(distances), transform(eps), 6
    )
    np.testing.assert_array_equal(transformed, expected)


@pytest.mark.parametrize("seed", range(3))
def test_reordering_the_points_reorders_the_clusters(seed: int) -> None:
    rng = np.random.default_rng(seed)
    neighbors, distances = random_graph(rng, 1000, 8)
    eps = eps_at_quantile(distances, 6, 0.6)
    labels = knn_dbscan_from_graph(neighbors, distances, eps, 6)
    order = rng.permutation(len(neighbors))  # new point j is old point order[j]
    position = np.argsort(order)
    permuted = knn_dbscan_from_graph(
        position[neighbors[order]], distances[order], eps, 6
    )
    np.testing.assert_array_equal(permuted == -1, labels[order] == -1)
    assert same_partition(permuted, labels[order])


def test_columns_beyond_min_samples_minus_one_are_never_read() -> None:
    rng = np.random.default_rng(3)
    neighbors, distances = random_graph(rng, 1500, 30)
    for min_samples in (2, 6, 15, 31):
        eps = eps_at_quantile(distances, min_samples, 0.5)
        np.testing.assert_array_equal(
            knn_dbscan_from_graph(neighbors, distances, eps, min_samples),
            knn_dbscan_from_graph(
                neighbors[:, : min_samples - 1],
                distances[:, : min_samples - 1],
                eps,
                min_samples,
            ),
        )


def _clique(distance: np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Six points, each the others' neighbors at one shared float32 distance."""
    neighbors = np.array([[j for j in range(6) if j != i] for i in range(6)])
    return neighbors, np.full(neighbors.shape, distance, dtype=np.float32)


def test_a_distance_equal_to_eps_is_within_eps() -> None:
    d = np.float32(0.3)
    graph = _clique(d)
    np.testing.assert_array_equal(knn_dbscan_from_graph(*graph, float(d), 6), [0] * 6)
    below = float(np.nextafter(d, np.float32(0)))
    np.testing.assert_array_equal(knn_dbscan_from_graph(*graph, below, 6), [-1] * 6)


def test_eps_is_compared_in_float32() -> None:
    # A float64 eps just below 1.0 rounds to float32 1.0, which includes the grid's unit
    # distances; the float32 value just below 1.0 excludes them. The estimator's
    # core_sample_indices_ must agree with the library's decision both times.
    X = _grid()
    for eps, n_core in [
        (1.0, 100),
        (float(np.nextafter(1.0, 0.0)), 100),
        (float(np.nextafter(np.float32(1), np.float32(0))), 0),
    ]:
        model = KNNDBSCAN(eps, 5, device="cpu").fit(X)
        assert len(model.core_sample_indices_) == n_core
        assert (model.labels_[model.core_sample_indices_] >= 0).all()
        assert ((model.labels_ >= 0).sum() == 0) == (n_core == 0)  # fmt: skip


@pytest.mark.parametrize("exponent", [-120, 120])
def test_extreme_magnitudes_cluster_like_unit_scale(exponent: int) -> None:
    # Near the float32 limits (about 1e-36 and 1e36 here), still exact powers of two.
    rng = np.random.default_rng(5)
    neighbors, distances = random_graph(rng, 600, 6)
    eps = eps_at_quantile(distances, 5, 0.5)
    scale = np.float32(2.0**exponent)
    np.testing.assert_array_equal(
        knn_dbscan_from_graph(neighbors, distances * scale, eps * float(scale), 5),
        knn_dbscan_from_graph(neighbors, distances, eps, 5),
    )
