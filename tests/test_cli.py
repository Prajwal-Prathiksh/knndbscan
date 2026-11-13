"""The MPI command-line program must satisfy the definitions on any number of ranks.

Theorem 3.6 of the paper: the distributed (inexact) MST finds the same clusters of core
points for every partition of the points over ranks. Border points may join any
adjacent cluster, so they are checked for validity, not equality.

Runs only when KNNDBSCAN_CLI names the built program (`make test-cli` sets it); the MPI
launcher is KNNDBSCAN_MPIEXEC, or `mpiexec` on PATH.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from reference import (
    assert_matches_reference,
    eps_at_quantile,
    random_graph,
    reference_clustering,
    same_partition,
)
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

from knndbscan import knn_dbscan_from_graph

CLI = os.environ.get("KNNDBSCAN_CLI")
MPIEXEC = os.environ.get("KNNDBSCAN_MPIEXEC") or shutil.which("mpiexec")

pytestmark = [
    pytest.mark.cli,
    pytest.mark.skipif(
        not CLI or not MPIEXEC, reason="KNNDBSCAN_CLI or mpiexec not set"
    ),
]


def _write_graph(path: Path, neighbors: np.ndarray, distances: np.ndarray) -> None:
    """Write the CLI's text format: each row starts with the point itself at distance 0."""
    distances = np.asarray(distances, dtype=np.float32)  # 9 digits round-trip float32
    with path.open("w") as file:
        for i, (row, dist) in enumerate(zip(neighbors, distances, strict=True)):
            pairs = " ".join(f"{d:.9g} {j}" for d, j in zip(dist, row, strict=True))
            file.write(f"{i} 0 {i} {pairs}\n")


def _run(path: Path, n_points: int, n_columns: int, eps: float, min_samples: int, *, ranks: int, threads: int = 2) -> np.ndarray:  # fmt: skip
    output = path.with_suffix(f".{ranks}r{threads}t.labels")
    run = subprocess.run(
        [MPIEXEC, "-n", str(ranks), CLI, "-n", str(n_points), "-e", repr(eps),
         "-m", str(min_samples), "-k", str(n_columns + 1), "-t", str(threads),
         "-i", str(path), "-o", str(output)],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert run.returncode == 0, f"mpiexec failed:\n{run.stdout}\n{run.stderr}"
    labels = np.loadtxt(output, dtype=np.int64, ndmin=1)
    assert len(labels) == n_points, f"{len(labels)} labels for {n_points} points"
    return labels


def _check(path, neighbors, distances, eps, min_samples, **run) -> np.ndarray:
    labels = _run(path, *neighbors.shape, eps, min_samples, **run)
    reference = reference_clustering(neighbors, distances, eps, min_samples)
    assert_matches_reference(labels, reference, border="any")
    library = knn_dbscan_from_graph(neighbors, distances, eps, min_samples)
    core = reference.core
    assert same_partition(labels[core], library[core])
    return labels


def _blobs_graph(order: str) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(n_samples=1500, centers=5, cluster_std=0.6, random_state=3)
    X[1::50] = X[0::50]  # duplicate points
    # Grouped by cluster, ranks own whole clusters and few edges cross ranks; shuffled,
    # most edges are cut edges and the distributed (cut) MST does most of the work.
    rng = np.random.default_rng(0)
    X = (
        X[np.argsort(y, kind="stable")]
        if order == "grouped"
        else X[rng.permutation(len(X))]
    )
    distances, neighbors = NearestNeighbors(n_neighbors=12).fit(X).kneighbors()
    return neighbors, distances


@pytest.mark.parametrize("order", ["shuffled", "grouped"])
@pytest.mark.parametrize("ranks", [1, 2, 3, 4, 6])
def test_data_graph_on_any_rank_count(tmp_path: Path, order: str, ranks: int) -> None:
    neighbors, distances = _blobs_graph(order)
    _write_graph(tmp_path / "graph.txt", neighbors, distances)
    _check(tmp_path / "graph.txt", neighbors, distances, 0.4, 8, ranks=ranks)


# Random graphs found a cycle-breaking hang on one particular split (5 ranks), so many
# splits are tried.
@pytest.mark.parametrize("ranks", [1, 2, 3, 4, 5, 7])
@pytest.mark.parametrize("seed", range(4))
def test_random_directed_graph_with_cycles_across_ranks(tmp_path: Path, ranks: int, seed: int) -> None:  # fmt: skip
    rng = np.random.default_rng(seed)
    neighbors, distances = random_graph(rng, 900, 8)
    eps = eps_at_quantile(distances, 7, 0.6)
    _write_graph(tmp_path / "graph.txt", neighbors, distances)
    _check(tmp_path / "graph.txt", neighbors, distances, eps, 7, ranks=ranks)


def test_ranks_that_own_no_points(tmp_path: Path) -> None:
    # 7 points over 6 ranks: blocks of 2 leave ranks 4 and 5 without points.
    X = np.array([0.0, 0.1, 0.2, 5.0, 5.1, 5.2, 20.0])[:, None]
    distances, neighbors = NearestNeighbors(n_neighbors=3).fit(X).kneighbors()
    _write_graph(tmp_path / "graph.txt", neighbors, distances)
    labels = _check(tmp_path / "graph.txt", neighbors, distances, 0.25, 3, ranks=6)
    assert same_partition(labels, [0, 0, 0, 1, 1, 1, -1])


def test_distances_above_ten_thousand(tmp_path: Path) -> None:
    neighbors, distances = _blobs_graph("shuffled")
    scale = 2.0**16
    _write_graph(tmp_path / "graph.txt", neighbors, distances * scale)
    labels = _check(
        tmp_path / "graph.txt", neighbors, distances * scale, 0.4 * scale, 8, ranks=3
    )
    assert len(np.unique(labels[labels >= 0])) == 5


def test_runs_repeat_identically_across_threads(tmp_path: Path) -> None:
    neighbors, distances = _blobs_graph("shuffled")
    _write_graph(tmp_path / "graph.txt", neighbors, distances)
    runs = [
        _run(tmp_path / "graph.txt", *neighbors.shape, 0.4, 8, ranks=3, threads=threads)
        for threads in (1, 4, 4, 4)
    ]
    for labels in runs[1:]:
        np.testing.assert_array_equal(labels == -1, runs[0] == -1)
        assert same_partition(labels, runs[0])


@pytest.mark.parametrize("level", ["off", "info", "debug"])
def test_log_level_selects_what_reaches_standard_error(
    tmp_path: Path, level: str
) -> None:
    neighbors, distances = _blobs_graph("shuffled")
    _write_graph(tmp_path / "graph.txt", neighbors, distances)
    run = subprocess.run(
        [MPIEXEC, "-n", "2", CLI, "-n", str(len(neighbors)), "-e", "0.4", "-m", "8",
         "-k", "13", "-t", "1", "-l", level,
         "-i", str(tmp_path / "graph.txt"), "-o", str(tmp_path / "labels.txt")],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert run.returncode == 0, run.stderr
    assert run.stdout == ""
    lines = run.stderr.splitlines()
    if level == "off":
        assert lines == []
    else:
        assert any(
            line.startswith("[knndbscan info] rank 0/2: cut MST:") for line in lines
        )
        has_debug = any(
            line.startswith("[knndbscan debug] rank 1/2:") for line in lines
        )
        assert has_debug == (level == "debug")
