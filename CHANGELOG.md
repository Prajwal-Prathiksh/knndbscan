# Changelog

User-visible changes to knndbscan, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may break the API).
Each release's GitHub release notes are its section below.

## [Unreleased]

### Added

- Prebuilt wheels for Windows on x86_64 and ARM64, for every Python from 3.10 (from 3.11
  on ARM64, the first Python with NumPy, SciPy and scikit-learn wheels for it).
- A technical companion to the paper, [docs/report/knndbscan.pdf](docs/report/knndbscan.pdf),
  which explains the method, its theorems, the distributed algorithm and how the
  implementation is verified, on one running example, without needing the paper first.

## [0.2.0] - 2026-11-13

A rebuild of the package around the original SC20 implementation, with fixes to its
clustering results.

### Added

- Prebuilt wheels for Linux (x86_64 and aarch64, glibc 2.28+) and macOS 26+ (Apple Silicon
  and Intel). One wheel per platform serves every Python from 3.10, free-threaded builds
  included, with no MPI, compiler or CUDA needed.
- `knn_dbscan_from_graph`, to cluster any precomputed kNN graph, exact or approximate.
- `KNNDBSCAN`, a scikit-learn estimator with `labels_` and `core_sample_indices_`.
- Neighbor search on NVIDIA GPUs with `device="gpu"` (the `gpu` extra: CuPy and cuML).
- Progress logging through Python's `logging` module, on the `knndbscan` logger; the
  command-line program takes `-l off|info|debug`.
- A C API (`knndbscan.h`) for C and C++ programs, and a CMake build of the MPI
  command-line program (`make cli`).

### Changed

- A distance exactly equal to `eps` counts as within `eps`, as in the paper and in
  scikit-learn's DBSCAN, so core, border and noise points are exactly DBSCAN's.
- Clusters are numbered in order of their lowest-indexed point, so labels are identical
  for any number of threads.
- Invalid input raises `ValueError` naming the first bad row (index out of range, a point
  listed as its own neighbor, unsorted or negative distances, `min_samples` out of range)
  instead of crashing, hanging or returning wrong labels.
- **Breaking:** `run_knndbscan` is replaced by `knn_dbscan_from_graph`, which takes 2-D
  `(n_points, n_neighbors)` arrays; `mpi_threads` is `n_jobs`; `knn_dbscan` no longer
  takes `k_neighbors` (graph columns beyond `min_samples - 1` are never used) or
  `verbose` (use logging).
- Licensed under BSD-3-Clause.

### Fixed

- Clusters whose distances were 10,000 or more were never merged.
- On several MPI ranks, clusters could be split (for example one cluster became six on
  seven ranks) or the run could hang, when the minimum edges between subtrees formed a
  cycle.
- With several threads, subtrees caught in a cycle of minimum edges could be merged into
  an unrelated cluster, and core points could be given another thread's cluster.
- The command-line program could write an empty labels file when run on several ranks.

### Removed

- The runtime dependency on MPI, and on `matplotlib` and `pip`.

[Unreleased]: https://github.com/Prajwal-Prathiksh/knndbscan/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Prajwal-Prathiksh/knndbscan/releases/tag/v0.2.0
