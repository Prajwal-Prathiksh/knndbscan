# knndbscan

Density-based clustering of kNN graphs: a Python package and C library implementing
kNN-DBSCAN ([Chen, Ruys and Biros](https://arxiv.org/abs/2009.04552)), with a parallel C++
core (OpenMP threads, and MPI ranks for the command-line program).

kNN-DBSCAN works from a k-nearest-neighbor graph instead of range queries, so it scales to
high-dimensional data where DBSCAN's neighborhoods become expensive, and it accepts any kNN
graph, exact or approximate.

## Installation

```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ knndbscan
```

| | |
|---|---|
| Python | 3.10 and later, including free-threaded builds. One wheel per platform serves every version. |
| Platforms | Linux (x86_64, aarch64; glibc 2.28+) and macOS 15+ (Apple Silicon and Intel). |
| Dependencies | NumPy and scikit-learn. No MPI, compiler or CUDA needed. |
| GPU neighbor search | `pip install "knndbscan[gpu]"` adds CuPy and cuML (CUDA 12, Linux). |

With `uv`, keep the other packages on PyPI by pinning only knndbscan to TestPyPI:

```toml
[[tool.uv.index]]
name = "testpypi"
url = "https://test.pypi.org/simple/"
explicit = true

[tool.uv.sources]
knndbscan = { index = "testpypi" }
```

## Quick start

```python
from sklearn.datasets import make_moons

from knndbscan import KNNDBSCAN, knn_dbscan, knn_dbscan_from_graph

X, _ = make_moons(n_samples=100_000, noise=0.06, random_state=0)

labels = knn_dbscan(X, eps=0.05, min_samples=15, n_jobs=-1)  # -1 for noise

model = KNNDBSCAN(eps=0.05, min_samples=15, n_jobs=-1).fit(X)  # scikit-learn estimator
model.labels_, model.core_sample_indices_

# Or bring your own kNN graph (exact, or approximate from FAISS, pynndescent, ...):
# rows sorted by distance, each point excluded from its own row.
from sklearn.neighbors import NearestNeighbors

distances, neighbors = NearestNeighbors(n_neighbors=14).fit(X).kneighbors()
labels = knn_dbscan_from_graph(neighbors, distances, eps=0.05, min_samples=15)
```

## How the results relate to DBSCAN

- **Core, border and noise points are DBSCAN's.** A point is a core point when its
  `(min_samples - 1)`-th nearest neighbor (the point itself counts towards `min_samples`,
  as in scikit-learn) lies at a distance `<= eps`.
- **Clusters can differ.** Two core points are linked only through kNN edges: one must be
  among the other's `min_samples - 1` nearest neighbors. kNN-DBSCAN can therefore split a
  cluster that DBSCAN joins through a sparse bridge, but never joins clusters that DBSCAN
  separates; with the same number of clusters, the two agree (Theorem 3.4 of the paper).
- **A wider graph is reusable.** Columns beyond `min_samples - 1` are not used, so a graph
  with `k` neighbors serves every `min_samples <= k + 1` without recomputing it.
- **Labels** are `0..n_clusters-1`, numbered by each cluster's lowest-indexed point, and
  `-1` for noise. They do not depend on the number of threads.

`examples/compare_with_dbscan.py` runs both algorithms side by side.

## Logging

knndbscan reports progress through Python's standard `logging`, on the `knndbscan`
logger: the neighbor search (`knndbscan.neighbors`) and each phase of the C++ core
(`knndbscan.core`: core points, subtrees, clusters, timings) at INFO, and every Borůvka
step at DEBUG. It is silent until you enable it:

```python
import logging

logging.basicConfig(level=logging.INFO)  # everything, INFO and up
logging.getLogger("knndbscan").setLevel(logging.DEBUG)  # or only knndbscan, in detail
```

The level is read on every call, and disabled messages are never formatted, so logging
costs nothing when off. The command-line program logs the same messages (`-l` below),
prefixed with the MPI rank.

## Command-line program (MPI)

For graphs too large for one machine, the command-line program splits the rows across MPI
ranks. It needs an MPI installation (`mpicxx`, `mpiexec`):

```bash
make cli                                    # builds build/cli/knndbscan
mpiexec -n 4 build/cli/knndbscan -n 70000 -e 1300.0 -m 100 -k 100 -t 4 \
    -i mnist70k.knn.txt -o labels.txt
```

| Flag | Meaning |
|---|---|
| `-n` | number of points |
| `-e` | `eps` |
| `-m` | `min_samples` |
| `-k` | neighbors per line of the input file, including the point itself (at least `-m`) |
| `-t` | OpenMP threads per rank |
| `-i`, `-o` | input graph file, output labels file (one label per line) |
| `-l` | log level on standard error: `off`, `info` (default) or `debug`; also `$KNNDBSCAN_LOG_LEVEL` |

Each input line is `point_id distance_1 neighbor_1 distance_2 neighbor_2 ...`, sorted by
distance and **including the point itself** at distance 0, for example `7 0.0 7 1.0 10 2.0 11`.
The program drops the point from its row, so its results equal the Python package's on the
same graph (cluster numbering aside).

## Development

```bash
make install      # editable install (uv sync); rebuilds the C library when cpp/ changes
make check        # lint, Python and C/C++ tests
make check-all    # also the MPI program on 1-7 ranks and the sanitizer builds (needs MPI)
make              # every target, with a description
```

[AGENTS.md](AGENTS.md) describes the code layout, conventions and verification steps;
[docs/ALGORITHM.md](docs/ALGORITHM.md) how the code implements the paper;
[docs/RELEASING.md](docs/RELEASING.md) the release process.

## License

BSD 3-Clause; see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{chen2024knndbscandbscanhighdimensions,
      title={KNN-DBSCAN: a DBSCAN in high dimensions},
      author={Youguang Chen and William Ruys and George Biros},
      year={2024},
      eprint={2009.04552},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2009.04552},
}
```

The algorithm core is the SC20 implementation from
[ut-padas/knndbscan](https://github.com/ut-padas/knndbscan);
[docs/sc20_environment.txt](docs/sc20_environment.txt) records the environment of the
paper's runs.
