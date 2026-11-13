"""kNN-DBSCAN computed directly from the definitions of Chen, Ruys and Biros (arXiv:2009.04552).

The tests compare the C++ implementation against this module, so it must stay an
independent statement of the paper, never a transcription of cpp/core:

- Lemma 3.1: a point is core when its (min_samples - 1)-th neighbor is within eps (<=);
  a non-core point is a border point when a core point within eps is among its first
  min_samples - 1 neighbors, and noise otherwise.
- Definition 3.6 and Theorem 3.5: clusters of core points are the connected components
  of the core points under "q is among p's first min_samples - 1 neighbors, or p among
  q's" (the directed kNN graph, symmetrized).
- Definition 3.9: a border point may join any cluster of a core point within eps among
  its neighbors; the library takes the first such neighbor in row order.

Distances and eps are compared in float32, the precision of the C library.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class Reference:
    """Per point: core flag, cluster of a core point (-1 otherwise), and border candidates."""

    core: np.ndarray
    component: np.ndarray
    # For border points, the components of their core neighbors within eps, nearest
    # first; empty for core and noise points.
    candidates: list[np.ndarray]

    @property
    def noise(self) -> np.ndarray:
        return ~self.core & np.array([len(c) == 0 for c in self.candidates])


def reference_clustering(
    neighbors: np.ndarray, distances: np.ndarray, eps: float, min_samples: int
) -> Reference:
    m = min_samples - 1
    neighbors = np.asarray(neighbors)[:, :m]
    distances = np.asarray(distances, dtype=np.float32)[:, :m]
    within = distances <= np.float32(eps)
    n_points = len(neighbors)

    core = within[:, m - 1]
    rows = np.repeat(np.arange(n_points), m)
    cols = neighbors.ravel()
    linked = core[rows] & core[cols]
    graph = coo_matrix(
        (np.ones(linked.sum()), (rows[linked], cols[linked])),
        shape=(n_points, n_points),
    )
    _, components = connected_components(graph, directed=True, connection="weak")
    component = np.where(core, components, -1)

    reaches_core = within & core[neighbors]
    candidates = [
        component[neighbors[p][reaches_core[p]]] if not core[p] else np.empty(0, int)
        for p in range(n_points)
    ]
    return Reference(core, component, candidates)


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True when a and b group the same elements together, whatever the label values."""
    pairs = set(zip(np.asarray(a).tolist(), np.asarray(b).tolist(), strict=True))
    return (
        len(pairs)
        == len(set(np.asarray(a).tolist()))
        == len(set(np.asarray(b).tolist()))
    )


def assert_matches_reference(
    labels: np.ndarray, reference: Reference, *, border: str = "nearest"
) -> None:
    """Check noise exactly, core clusters as a partition, and each border point's cluster.

    border="nearest" requires the cluster of the first core neighbor within eps (the
    library's rule); border="any" accepts any such neighbor's cluster, which is all the
    definitions require (the MPI program may pick a neighbor on its own rank).
    """
    labels = np.asarray(labels)
    core = reference.core
    np.testing.assert_array_equal(labels == -1, reference.noise, "noise points differ")
    assert same_partition(labels[core], reference.component[core]), (
        "core clusters differ"
    )
    label_of = dict(
        zip(reference.component[core].tolist(), labels[core].tolist(), strict=True)
    )
    for p in np.flatnonzero(~core & ~reference.noise):
        allowed = [label_of[c] for c in reference.candidates[p].tolist()]
        if border == "nearest":
            assert labels[p] == allowed[0], (
                f"border point {p}: {labels[p]} != {allowed[0]}"
            )
        else:
            assert labels[p] in allowed, (
                f"border point {p}: {labels[p]} not in {allowed}"
            )


def random_graph(
    rng: np.random.Generator,
    n_points: int,
    n_neighbors: int,
    *,
    levels: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A random directed graph in the library's layout: distinct non-self neighbors, sorted rows.

    Its distances satisfy no triangle inequality and its edges are rarely mutual, so
    Boruvka meets the cycles of a directed approximate kNN graph (Section 4.2 of the
    paper). With `levels`, distances take only that many values, forcing ties.
    """
    offsets = rng.random((n_points, n_points - 1)).argsort(axis=1)[:, :n_neighbors] + 1
    neighbors = (np.arange(n_points)[:, None] + offsets) % n_points
    distances = rng.random((n_points, n_neighbors))
    if levels is not None:
        distances = np.floor(distances * levels) / levels
    return neighbors, np.sort(distances, axis=1).astype(np.float32)


def eps_at_quantile(distances: np.ndarray, min_samples: int, quantile: float) -> float:
    """An eps equal to an actual (min_samples - 1)-th neighbor distance, so ties with eps occur."""
    column = np.asarray(distances, dtype=np.float32)[:, min_samples - 2]
    return float(np.quantile(column, quantile, method="lower"))
