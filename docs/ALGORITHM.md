# How the code implements kNN-DBSCAN

The algorithm is that of Chen, Ruys and Biros, "KNN-DBSCAN: a DBSCAN in high dimensions"
([arXiv:2009.04552](https://arxiv.org/abs/2009.04552)); section, algorithm and figure
numbers below refer to it. This document maps the paper onto `cpp/core/` and records the
invariants the code depends on. The definitions themselves are restated, independently of
the code, in `tests/reference.py`. For the method itself (definitions, theorems with
proof sketches, pseudocode, the distributed algorithm and how correctness is verified),
read the report [docs/report/knndbscan.pdf](report/knndbscan.pdf).

## What is computed

For a kNN graph (each point's neighbors, nearest first, without the point itself) and
parameters `eps` and `min_samples`:

1. A point is a **core point** when its `(min_samples - 1)`-th neighbor is within `eps`
   (`<=`; Lemma 3.1).
2. Core points are linked when one is among the other's first `min_samples - 1`
   neighbors (Definition 3.6, symmetrized); **clusters** are the connected components of
   that graph (Theorem 3.5). Graph columns beyond `min_samples - 1` are never read.
3. A non-core point with a core point within `eps` among those neighbors is a **border
   point** and joins one such core point's cluster; any other point is **noise**.

The clusters are found as the components of a spanning forest built by Borůvka's
algorithm, first inside each MPI rank and then across ranks (the "inexact MST" of
Section 3.4, which Theorem 3.6 shows gives the same clusters for any split of the points).

## Stages and files

`knndbscan()` in `knndbscan.cpp` runs the stages of Algorithm 1 in order. The core is one
translation unit: `knndbscan.cpp` includes every stage file.

| Stage | Paper | File | Function |
|---|---|---|---|
| I. Core points | Algorithm 1, lines 2-6 | `knndbscan.cpp` | `knndbscan` |
| II. Local MST | Algorithm 2 | `local_mst.cpp` | `local_mst` |
| III. Global subtree ids | Section 4.3 | `partition.h` | `label_subtrees` |
| IV. Cut edges relabelled | Algorithm 4, lines 3-5 | `cut_edges.cpp` | `update_cut_edges` |
| V. Cut MST | Algorithm 4 | `cut_mst.cpp` | `cut_mst` |
| VI. Border points | Algorithm 1, line 12 | `border_points.cpp` | `label_border_points` |

Shared pieces: `min_edges.h` (the minimum-edge update of Algorithm 3), `cycles.h`
(grouping subtrees caught in cycles of minimum edges), `partition.h` (which rank owns
which points and subtrees), `log.h` (logging), `globals.h` (types, global state, OpenMP
reductions, and the choice between real MPI and `cpp/mpi_serial.h`).

`cpp/capi.cpp` wraps `knndbscan()` for the library: it validates the graph, serializes
calls (the core's state is global), sets the OpenMP thread count and log level, and
renumbers clusters by their lowest-indexed point. `cpp/cli/knndbscan_cli.cpp` wraps it
for MPI runs.

## Distribution over ranks

- Points are split into contiguous blocks in rank order (`count_points`); a rank stores
  only the rows of its own points, with global neighbor indices.
- Edges to points of another rank are **cut edges**; the local MST ignores them and
  collects them in `crossE`.
- After the local MST, every subtree gets a global id; ids are contiguous per rank
  (`ilabels`), and the rank that created a subtree **owns** it (`jlabels`). A merged
  subtree keeps the id, and the owner, of its root.
- In each cut-MST step, every rank proposes the lightest cut edge it holds for each
  subtree it touches, and the owner keeps the lightest proposal (`minimum_edges`). Roots
  are then found by distributed pointer jumping (`find_roots`).

## Invariants

These hold the results together; each has a test that fails when it is broken.

- **Cycles of minimum edges.** With directed or approximate kNN graphs, minimum edges can
  form cycles longer than two (Fig. 4). Pointer jumping that updates all pointers at once
  turns a cycle whose length is a power of two into members that each point at
  themselves; such a member must stay unresolved (`post_recv`), not become a root. The
  unresolved subtrees are grouped with a union-find (`cycles.h`), never by overwriting
  representatives' pointers, which can itself create cycles.
- **No finite sentinel.** The "no edge yet" weight is infinite, so every finite edge can
  be selected whatever the scale of the distances.
- **Parallel loops write only their own iteration's data.** A variable written inside an
  `omp parallel for` is declared inside it.
- **The minimum-edge update is not atomic** (`min_edges.h`): threads may record an edge
  that is not the lightest. This changes the spanning tree, not the components.
- **Labels do not depend on threads or ranks.** The core's own numbering follows its
  roots, which depend on scheduling; `capi.cpp` renumbers by lowest-indexed point. The CLI
  writes the core's numbering.
- **Border points prefer their own rank.** `label_border_points` takes the first core
  neighbor within `eps` on the point's own rank, and otherwise asks other ranks, so on
  several ranks a border point may join a different (equally valid) adjacent cluster than
  on one.
