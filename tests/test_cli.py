"""The MPI command-line program must find the Python library's partition on any number of ranks.

Runs only when KNNDBSCAN_CLI names the built program (`make test-cli` sets it); the MPI
launcher is KNNDBSCAN_MPIEXEC, or `mpiexec` on PATH.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

from knndbscan import knn_dbscan_from_graph
from knndbscan.neighbors import _drop_self

CLI = os.environ.get("KNNDBSCAN_CLI")
MPIEXEC = os.environ.get("KNNDBSCAN_MPIEXEC") or shutil.which("mpiexec")

pytestmark = [
    pytest.mark.cli,
    pytest.mark.skipif(
        not CLI or not MPIEXEC, reason="KNNDBSCAN_CLI or mpiexec not set"
    ),
]

EPS = 0.4
MIN_SAMPLES = 8
GRAPH_COLUMNS = 12  # the file holds more neighbors than the CLI keeps


def _renumber_by_first_point(labels: np.ndarray) -> np.ndarray:
    """The CLI keeps the core's numbering; the library numbers clusters by first point."""
    in_cluster = labels >= 0
    _, first, inverse = np.unique(
        labels[in_cluster], return_index=True, return_inverse=True
    )
    renumbered = np.full(len(labels), -1)
    renumbered[in_cluster] = np.argsort(np.argsort(first))[inverse]
    return renumbered


@pytest.fixture(scope="module")
def graph_file(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
    X, _ = make_blobs(n_samples=1500, centers=5, cluster_std=0.6, random_state=3)
    X[1::50] = X[0::50]  # duplicates, so some rows do not start with the point itself
    distances, indices = (
        NearestNeighbors(n_neighbors=GRAPH_COLUMNS).fit(X).kneighbors(X)
    )
    # 9 significant digits round-trip float32, so the CLI parses the library's values.
    distances = distances.astype(np.float32)
    path = tmp_path_factory.mktemp("cli") / "graph.txt"
    with path.open("w") as file:
        for i, (row_indices, row_distances) in enumerate(
            zip(indices, distances, strict=True)
        ):
            pairs = " ".join(
                f"{d:.9g} {j}" for d, j in zip(row_distances, row_indices, strict=True)
            )
            file.write(f"{i} {pairs}\n")
    neighbors, distances = _drop_self(indices, distances)
    expected = knn_dbscan_from_graph(
        neighbors[:, : MIN_SAMPLES - 1],
        distances[:, : MIN_SAMPLES - 1],
        EPS,
        MIN_SAMPLES,
    )
    return path, expected


@pytest.mark.parametrize("n_ranks", [1, 2, 3])
def test_cli_matches_library_on_any_rank_count(
    graph_file: tuple[Path, np.ndarray], n_ranks: int, tmp_path: Path
) -> None:
    path, expected = graph_file
    output = tmp_path / "labels.txt"
    run = subprocess.run(
        [MPIEXEC, "-n", str(n_ranks), CLI, "-n", str(len(expected)), "-e", str(EPS),
         "-m", str(MIN_SAMPLES), "-k", str(GRAPH_COLUMNS), "-t", "2",
         "-i", str(path), "-o", str(output)],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert run.returncode == 0, f"mpiexec failed:\n{run.stdout}\n{run.stderr}"
    labels = np.loadtxt(output, dtype=np.int64)
    np.testing.assert_array_equal(_renumber_by_first_point(labels), expected)
