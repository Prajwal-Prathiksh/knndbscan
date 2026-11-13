# AGENTS.md

Guidance for coding agents working in this repository (knndbscan: the parallel kNN-DBSCAN
algorithm of Chen, Ruys and Biros, as a C library with a Python package and an MPI
command-line program).

## Environment

- Python environment: `.venv/`, managed by `uv` (`uv sync` installs the package and the
  `dev` group). The development Python is pinned in `.python-version`; the package itself
  supports every version allowed by `requires-python`.
- `uv sync` rebuilds the C library whenever a file under `cpp/` or `CMakeLists.txt`
  changes (`tool.uv.cache-keys`). CMake and Ninja come from the `dev` group, so no system
  install is needed.
- The Python package never uses MPI. MPI is needed only for the command-line program
  (`make cli`), which finds `mpicxx` through CMake's `FindMPI`.
- Tests and benchmarks pass `device="cpu"` or `device="gpu"` explicitly, so a missing GPU
  cannot turn a GPU test into a CPU one.

## Verification

Run from the repository root:

```bash
make test        # pytest on the editable install
make lint        # ruff, mypy, clang-format --dry-run
make test-cli    # builds the CLI against MPI; compares 1 and 2 ranks with the Python result
make test-wheel  # builds the wheel and runs the tests from it on every supported Python
```

- `make test-wheel` is the check that matters for packaging: it installs the built wheel
  (not the source tree) into a fresh environment per Python version, with both the newest
  and the oldest allowed dependency versions. Run it after any change to `pyproject.toml`,
  `CMakeLists.txt`, `cpp/capi.cpp` or `src/knndbscan/_native.py`.
- Clustering results must not depend on the thread count or the number of MPI ranks;
  `tests/test_cluster.py` and `tests/test_cli.py` check both. Treat any difference as a bug.
- A change to `cpp/core/` must leave the labels of the whole test suite unchanged unless
  the change is meant to alter them; say so and justify it in the commit message.

## Where things are

| Path | Contents |
|---|---|
| `cpp/include/knndbscan.h` | The public C API: the only symbols the shared library exports. |
| `cpp/capi.cpp` | The C API implementation: input validation, the call lock, error messages. |
| `cpp/mpi_serial.h` | A single-process implementation of the MPI calls the core makes. |
| `cpp/core/` | The kNN-DBSCAN algorithm (Borůvka MST over the kNN graph, distributed with MPI, threaded with OpenMP). |
| `cpp/cli/knndbscan_cli.cpp` | Command-line program: reads a kNN-graph text file and runs on any number of MPI ranks. |
| `src/knndbscan/` | The Python package: `_native.py` (ctypes binding), `cluster.py` (public API), `neighbors.py` (kNN graphs on CPU or GPU). |
| `tests/`, `examples/` | pytest suite; runnable examples (`uv run python examples/<name>.py`). |
| `docs/` | Documentation other than README.md and AGENTS.md (releasing, SC20 environment). |

- `src/knndbscan/__init__.py` re-exports the public API; import from `knndbscan`, not from
  the module a name lives in.
- The Python package talks to C only through `_native.py`. Do not add a second binding or
  a Python C-API extension: the wheel is tagged `py3-none-<platform>` and installs on every
  Python 3 because the library does not link against Python or NumPy.

Names (use them in code, docstrings, messages and docs):

- **kNN graph**: for each of the `n_points` points, its `n_neighbors` nearest neighbors in
  ascending order of distance, **excluding the point itself**, as two row-major arrays
  `neighbors` (int32) and `distances` (float32) of shape `(n_points, n_neighbors)`.
- **min_samples** counts the point itself, as in scikit-learn. A point is a **core point**
  when the distance to its `(min_samples - 1)`-th neighbor is strictly less than `eps`.
- Two core points are in the same cluster when they are connected through kNN-graph edges
  between core points among each one's first `min_samples - 1` neighbors. This is
  kNN-DBSCAN, not DBSCAN: it can split a cluster that DBSCAN would join, and extra
  neighbors beyond `min_samples - 1` only change how border points are assigned.
- **Labels**: `0..n_clusters-1` for clusters, `-1` for noise.

## The algorithm core (`cpp/core/`)

`cpp/core/` is the SC20 implementation from ut-padas/knndbscan. It is a unity build:
`clusters.cpp` includes the other `.cpp` files, and its state lives in globals declared in
`globals.h`. That shapes how it may be changed:

- Keep changes to `cpp/core/` minimal and semantic only where required. Input checks,
  threading policy and error reporting belong in `cpp/capi.cpp`, which is the only caller
  of `knndbscan()` besides the CLI.
- The globals make `knndbscan()` non-reentrant. `cpp/capi.cpp` serialises calls with a
  mutex; keep every entry point behind it.
- The core indexes the graph with 32-bit `int` products (`p * maxk`), so
  `n_points * n_neighbors` must stay below 2^31; `cpp/capi.cpp` rejects larger inputs.
- A new MPI call in the core must also be implemented in `cpp/mpi_serial.h`, with the
  semantics of a one-rank communicator, or the Python library stops building.

## Code changes

- Before adding a helper, search for an existing one and reuse or extend it. Do not add
  small single-use functions, and do not duplicate logic a module already owns (input
  validation lives in `cpp/capi.cpp`, graph construction in `neighbors.py`).
- A name another module imports or calls is part of its module's API: no leading
  underscore. `_name` stays inside its module (and its class); tests may still reach in.
- When existing code gains new behaviour, rename the function, its docstring and related
  identifiers to describe what it now does, and update every caller, test and doc.
- Add or update tests with every behaviour change.
- Dependencies get lower bounds only (`numpy>=1.24`), never upper caps, and
  `requires-python` has no upper bound. Supporting a new Python version means adding it to
  the CI matrix, the `test-wheel` target and the classifiers.

Why: reusing and extending shared modules, with tests alongside, solves several use cases
in fewer lines, and upper bounds break users' resolvers long before they protect them.

## Performance changes

Optimizations cost reading time, invariants a later edit can break, and new failure modes;
they must buy a measured improvement that matters.

- **Correctness first.** Never trade the independence of results from thread or rank
  count for speed.
- **Show it, do not assume it.** On a workload that matters (millions of points) with
  baseline time `t`, a speed change must save at least `max(2 s, min(5% of t, 10 s))`, or
  give a 10x speedup of a short run. A memory change must cut peak usage by at least 10%.
- **Measure properly:** interleaved A/B runs against the previous state, two or more
  rounds, and identical labels for changes claimed exact. Remove changes that do not meet
  the bar instead of keeping them behind a switch.

## Code documentation style

### Public API: Google style

Every name exported from `knndbscan/__init__.py` (functions, classes and their public
methods) gets a Google-style docstring: a one-line summary, then `Args:`, `Returns:`,
`Raises:` and, where useful, `Example:`. These docstrings are the user documentation, so
they state units, shapes, defaults and conventions (self-exclusion, `min_samples`
counting the point itself, strict `< eps`) even when the type hints imply them.

```python
def knn_dbscan_from_graph(neighbors, distances, eps, min_samples, *, n_jobs=None):
    """Cluster a precomputed kNN graph with kNN-DBSCAN.

    Args:
        neighbors: Neighbor indices, shape (n_points, n_neighbors), each row in ascending
            order of distance and excluding the point itself.
        ...

    Returns:
        Labels of shape (n_points,): 0..n_clusters-1 for clusters, -1 for noise.

    Raises:
        ValueError: If the graph is malformed (see the message for the first bad row).
    """
```

### Internal code: concise and contextual

Apply to all other Python code, and to existing code you edit once you understand it.

Type hints and descriptive names already state the parameters, types and return values;
docstrings should carry the context the code cannot express.

- Every module starts with a concise docstring stating its purpose.
- Every internal class, function and method gets a concise docstring; prefer one useful
  sentence. Skip it only when the name and a short body already say everything (a
  property returning a field, a test named after the behaviour it checks).
- Document what the signature and implementation do not say: contracts, invariants,
  design reasons, side effects, ownership, threading, performance limits, failure
  behaviour.
- Do not paraphrase the name, parameters, return annotation or visible code.
- Comment only where a local decision would otherwise be easy to misread or break, and
  explain why, not what the next line does.

```python
# Bad
def drop_self(indices, distances):
    """Drop self from the neighbors."""

# Good
def drop_self(indices, distances):
    """Remove each point from its own row; with duplicate points it may not be in column 0."""
```

### C and C++

- **File header:** one line naming what the file provides and who calls it.
- **Public C API** (`knndbscan.h`): each function documents its contract: ownership of
  buffers, return codes, thread safety, size limits. This is the reference for C users and
  for `_native.py`.
- **Internal functions:** document only what the code cannot show: which thread or rank
  owns what, why an order is fixed, what must not be "optimized away".

```cpp
// Bad: restates the call
// Lock the mutex.
std::lock_guard<std::mutex> lock(call_mutex);

// Good: the reason the line exists
// The core keeps its state in globals, so concurrent calls would corrupt each other.
std::lock_guard<std::mutex> lock(call_mutex);
```

### Committed documentation stands on its own

Before committing, make every comment, docstring, message and current-state doc (README,
AGENTS, docs/) describe the code as it is, in technical terms, readable without knowing how
it was built. History belongs in commit messages.

- No change narrative: "previously", "used to", "now", "renamed from".
- No pointers to reviews or chat sessions: state the reason itself. Cite a stable source
  (the paper, a library's documentation) only for what a comment cannot hold.
- No one-off measurements as decoration; keep a number only when it justifies a constant
  or a limit, and give the condition it depends on.
