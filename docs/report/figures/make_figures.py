"""Draw the figures of the report's visual appendix (docs/report/figures/*.svg).

Usage (from the repository root): uv run python docs/report/figures/make_figures.py

Every panel is computed, not drawn by hand: the kNN graph and point classes follow the
paper's definitions, Borůvka's rounds are replayed exactly as the report's Algorithm 1
states them, and the final clusterings come from the knndbscan library and from
scikit-learn's DBSCAN. A reproduction that disagrees with the library stops the script.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from knndbscan import knn_dbscan_from_graph

OUT = Path(__file__).resolve().parent
EPS, MIN_SAMPLES = 0.55, 6
PALETTE = ["#c0392b", "#e1a800", "#2e7d32", "#1f5fa8", "#6d3b0f", "#ef7d1a", "#6a1b9a",
           "#d81b8a", "#00838f", "#7cb342", "#5c6bc0", "#a1887f", "#f06292", "#26a69a"]  # fmt: skip
GREY, LIGHT, NOISE = "#9e9e9e", "#d6d6d6", "#bdbdbd"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "svg.fonttype": "path",
    # A fixed salt and no date make the SVGs identical from run to run.
    "svg.hashsalt": "knndbscan",
    "axes.linewidth": 0.6, "axes.edgecolor": "#555555",
})  # fmt: skip


@dataclass
class Graph:
    """A kNN graph in the library's layout, with each point's class under (EPS, MIN_SAMPLES)."""

    X: np.ndarray
    neighbors: np.ndarray
    distances: np.ndarray

    @property
    def core(self) -> np.ndarray:
        return self.distances[:, MIN_SAMPLES - 2] <= EPS

    def core_edges(self) -> list[tuple[int, int]]:
        """Edges of G_(M, core): each core point to its core neighbors among the first M-1."""
        core = self.core
        return [(i, int(j)) for i in np.flatnonzero(core)
                for j in self.neighbors[i, : MIN_SAMPLES - 1] if core[j]]  # fmt: skip


def make_data(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal((0.0, 0.0), 0.32, (34, 2))
    b = rng.normal((1.75, 0.15), 0.32, (34, 2))
    c = rng.normal((0.9, 2.15), 0.38, (30, 2))
    noise = rng.uniform((-1.0, -1.0), (2.9, 3.1), (9, 2))
    return np.vstack([a, b, c, noise])


def build(X: np.ndarray) -> Graph:
    distances, neighbors = (
        NearestNeighbors(n_neighbors=MIN_SAMPLES - 1).fit(X).kneighbors()
    )
    return Graph(X, neighbors, distances.astype(np.float32))


def choose_dataset() -> tuple[Graph, np.ndarray, np.ndarray]:
    """The first seed whose data shows a DBSCAN merge that kNN-DBSCAN keeps apart."""
    for seed in range(500):
        graph = build(make_data(seed))
        ours = knn_dbscan_from_graph(graph.neighbors, graph.distances, EPS, MIN_SAMPLES)
        theirs = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(graph.X)
        n_ours, n_theirs = ours.max() + 1, theirs.max() + 1
        border = (~graph.core & (ours >= 0)).sum()
        if n_ours == 3 and n_theirs == 2 and border >= 4 and (ours == -1).sum() >= 5:
            print(
                f"seed {seed}: kNN-DBSCAN {n_ours} clusters, DBSCAN {n_theirs}, {border} border"
            )
            return graph, ours, theirs
    raise SystemExit("no seed shows the DBSCAN/kNN-DBSCAN difference")


def boruvka_rounds(graph: Graph) -> list[dict]:
    """Replay Algorithm 1 (local Borůvka) and record each round's selections and labels."""
    core = graph.core
    members = np.flatnonzero(core)
    R = {int(i): int(i) for i in members}
    L = dict.fromkeys(R, 0)
    rounds = []
    while True:
        best: dict[
            int, tuple[float, int, int]
        ] = {}  # subtree -> (weight, proposer, target)
        for i in R:
            while L[i] < MIN_SAMPLES - 1:
                q, w = int(graph.neighbors[i, L[i]]), float(graph.distances[i, L[i]])
                if core[q] and R[q] != R[i]:
                    if R[i] not in best or (w, q) < best[R[i]][::2]:
                        best[R[i]] = (w, i, q)
                    break
                L[i] += 1
        subtrees = set(R.values())
        p = {u: (R[best[u][2]] if u in best else u) for u in subtrees}
        for u in subtrees:  # break symmetry between mutual pairs
            if p[p[u]] == u and u < p[u]:
                p[u] = u
        while True:  # pointer jumping
            jumped = {u: p[p[u]] for u in subtrees}
            if jumped == p:
                break
            p = jumped
        rounds.append(
            {"labels": dict(R), "selected": [(b[1], b[2]) for b in best.values()]}
        )
        R = {i: p[R[i]] for i in R}
        if len(set(R.values())) == len(subtrees):
            rounds.append({"labels": dict(R), "selected": []})
            return rounds


def colours_for(
    labels: dict[int, int], reference: dict[int, int] | None = None
) -> dict:
    """A colour per subtree; with `reference`, a subtree takes the colour of its final cluster."""
    roots = sorted(
        set(labels.values()), key=lambda r: min(i for i, s in labels.items() if s == r)
    )
    return {r: PALETTE[k % len(PALETTE)] for k, r in enumerate(roots)}


def panel(ax, title: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8.5, loc="left", pad=4)


def draw_edges(
    ax, X, edges, colour=LIGHT, width=0.6, arrows=False, zorder=1, alpha=1.0
):
    for i, j in edges:
        if arrows:
            ax.add_patch(FancyArrowPatch(X[i], X[j], arrowstyle="-|>", mutation_scale=7,
                                         color=colour, lw=width, shrinkA=2.5, shrinkB=2.5,
                                         zorder=zorder, alpha=alpha))  # fmt: skip
        else:
            ax.plot(*X[[i, j]].T, color=colour, lw=width, zorder=zorder, alpha=alpha)


def cluster_colours(labels: np.ndarray) -> list[str]:
    return [PALETTE[label % len(PALETTE)] if label >= 0 else NOISE for label in labels]


def figure_pipeline(graph: Graph, ours: np.ndarray, theirs: np.ndarray) -> None:
    X, core = graph.X, graph.core
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.9))
    directed = [
        (i, int(j))
        for i in range(len(X))
        for j in graph.neighbors[i, : MIN_SAMPLES - 1]
    ]

    ax = axes[0]
    panel(ax, r"(a) kNN graph: $M - 1 = 5$ out-edges")
    draw_edges(ax, X, directed, colour="#b0b0b0", width=0.45, arrows=True, alpha=0.8)
    ax.scatter(*X.T, s=9, c="#424242", zorder=3)

    ax = axes[1]
    panel(ax, "(b) core, border and noise")
    border = ~core & (ours >= 0)
    examples = [
        np.flatnonzero(core)[5],
        np.flatnonzero(border)[0],
        np.flatnonzero(ours == -1)[0],
    ]
    for i, colour in zip(examples, ["#1f5fa8", "#e1a800", "#757575"], strict=True):
        ax.add_patch(
            Circle(X[i], EPS, fill=False, ls="--", lw=0.7, color=colour, zorder=2)
        )
    ax.scatter(*X[core].T, s=12, c="#1f5fa8", zorder=3, label="core")
    ax.scatter(
        *X[border].T,
        s=14,
        c="#e1a800",
        edgecolors="#6d4c00",
        lw=0.5,
        zorder=3,
        label="border",
    )
    ax.scatter(*X[ours == -1].T, s=14, marker="x", c="#757575", zorder=3, label="noise")
    ax.legend(
        loc="upper left",
        fontsize=6.5,
        frameon=False,
        handletextpad=0.2,
        borderaxespad=0.2,
    )

    ax = axes[2]
    panel(ax, r"(c) $G_{M,\mathrm{core}}$ and its components")
    edges = graph.core_edges()
    draw_edges(ax, X, edges, colour="#7f7f7f", width=0.7)
    ax.scatter(*X[~core].T, s=6, c=LIGHT, zorder=2)
    ax.scatter(
        *X[core].T,
        s=13,
        c=[cluster_colours(ours)[i] for i in np.flatnonzero(core)],
        zorder=3,
    )

    ax = axes[3]
    panel(ax, "(d) DBSCAN joins two of them")
    matched = np.full_like(theirs, -1)
    for cluster in np.unique(theirs[theirs >= 0]):
        inside = ours[(theirs == cluster) & (ours >= 0)]
        matched[theirs == cluster] = np.bincount(inside).argmax()
    ax.scatter(*X.T, s=13, c=cluster_colours(matched), zorder=3)
    pairs = [(i, j) for i in np.flatnonzero(core) for j in np.flatnonzero(core)
             if i < j and ours[i] != ours[j] and np.linalg.norm(X[i] - X[j]) <= EPS]  # fmt: skip
    draw_edges(ax, X, pairs, colour="#c0392b", width=1.4, zorder=2)
    for axis in axes:
        axis.set_xlim(X[:, 0].min() - 0.35, X[:, 0].max() + 0.35)
        axis.set_ylim(X[:, 1].min() - 0.35, X[:, 1].max() + 0.35)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "pipeline.svg", bbox_inches="tight", metadata={"Date": None})


def figure_boruvka(graph: Graph, ours: np.ndarray) -> None:
    rounds = boruvka_rounds(graph)
    final = rounds[-1]["labels"]
    final_ids = {r: k for k, r in enumerate(sorted(set(final.values()), key=lambda r: min(
        i for i, s in final.items() if s == r)))}  # fmt: skip
    # The replay must find exactly the library's clusters.
    replay = np.full(len(graph.X), -1)
    for i, r in final.items():
        replay[i] = final_ids[r]
    core = graph.core
    assert np.array_equal(replay[core], ours[core]), (
        "the Borůvka replay disagrees with the library"
    )

    shown = rounds[:3] + [rounds[-1]]
    fig, axes = plt.subplots(1, len(shown), figsize=(10.2, 2.9))
    X = graph.X
    for k, (ax, state) in enumerate(zip(axes, shown, strict=True)):
        labels = state["labels"]
        colours = colours_for(labels)
        n = len(set(labels.values()))
        title = (
            f"(d) end: {n} subtrees"
            if k == len(shown) - 1
            else f"({'abc'[k]}) round {k + 1}: {n} subtrees"
        )
        panel(ax, title)
        draw_edges(ax, X, graph.core_edges(), colour="#ececec", width=0.5)
        same = [(i, j) for i, j in graph.core_edges() if labels[i] == labels[j]]
        for i, j in same:
            ax.plot(
                *X[[i, j]].T, color=colours[labels[i]], lw=1.1, alpha=0.55, zorder=1
            )
        for i, j in state["selected"]:
            ax.add_patch(FancyArrowPatch(X[i], X[j], arrowstyle="-|>", mutation_scale=8, color="#212121",
                                         lw=0.9, shrinkA=2.5, shrinkB=2.5, zorder=4))  # fmt: skip
        idx = np.array(sorted(labels))
        ax.scatter(
            *X[idx].T,
            s=15,
            c=[colours[labels[i]] for i in idx],
            edgecolors="white",
            lw=0.3,
            zorder=3,
        )
        ax.scatter(*X[~core].T, s=5, c=LIGHT, zorder=2)
        ax.set_xlim(X[:, 0].min() - 0.35, X[:, 0].max() + 0.35)
        ax.set_ylim(X[:, 1].min() - 0.35, X[:, 1].max() + 0.35)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "boruvka.svg", bbox_inches="tight", metadata={"Date": None})
    print("Borůvka rounds:", [len(set(r["labels"].values())) for r in rounds])


def limits(ax, X, pad=0.35) -> None:
    ax.set_xlim(X[:, 0].min() - pad, X[:, 0].max() + pad)
    ax.set_ylim(X[:, 1].min() - pad, X[:, 1].max() + pad)


def components(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    rows, cols = (np.array([e[k] for e in edges], dtype=int) for k in (0, 1))
    graph = coo_matrix((np.ones(len(edges)), (rows, cols)), shape=(n, n))
    return connected_components(graph, directed=True, connection="weak")[1]


def figure_distributed(graph: Graph, ours: np.ndarray, n_ranks: int = 3) -> None:
    # Number the points left to right, so that each rank's contiguous block is a vertical strip.
    order = np.argsort(graph.X[:, 0])
    position = np.argsort(order)
    X = graph.X[order]
    neighbors = position[graph.neighbors[order]]
    core = graph.core[order]
    final = ours[order]
    n0 = -(-len(X) // n_ranks)
    rank = np.arange(len(X)) // n0
    edges = [
        (i, int(j))
        for i in np.flatnonzero(core)
        for j in neighbors[i, : MIN_SAMPLES - 1]
        if core[j]
    ]
    local = [(i, j) for i, j in edges if rank[i] == rank[j]]
    cut = [(i, j) for i, j in edges if rank[i] != rank[j]]
    subtree = components(len(X), local)  # stage II: local components = local subtrees
    core_ids = np.flatnonzero(core)
    subtree_ids = sorted({int(subtree[i]) for i in core_ids})
    colour_of = {s: PALETTE[k % len(PALETTE)] for k, s in enumerate(subtree_ids)}

    # Stage V: components of the graph of subtrees joined by cut edges are the clusters.
    index = {s: k for k, s in enumerate(subtree_ids)}
    contracted = components(
        len(subtree_ids),
        [(index[int(subtree[i])], index[int(subtree[j])]) for i, j in cut],
    )
    merged = np.full(len(X), -1)
    for i in core_ids:
        merged[i] = contracted[index[int(subtree[i])]]
    pairs = set(zip(merged[core], final[core], strict=True))
    assert len(pairs) == len(set(final[core])) == len(set(merged[core])), (
        "Theorem 3.6 check failed"
    )

    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.9))
    bounds = [X[rank == r, 0].max() for r in range(n_ranks - 1)]
    tints = ["#e3ecf7", "#fbeedd", "#e6f2e3"]
    for ax in axes:
        for r in range(n_ranks):
            lo = (
                X[:, 0].min() - 0.35
                if r == 0
                else (bounds[r - 1] + X[rank == r, 0].min()) / 2
            )
            hi = (
                X[:, 0].max() + 0.35
                if r == n_ranks - 1
                else (bounds[r] + X[rank == r + 1, 0].min()) / 2
            )
            ax.axvspan(lo, hi, color=tints[r], zorder=0, lw=0)
        limits(ax, X)

    ax = axes[0]
    panel(ax, "(a) 3 ranks; cut edges in red")
    draw_edges(ax, X, local, colour="#8a8a8a", width=0.6)
    draw_edges(ax, X, cut, colour="#c0392b", width=1.1)
    ax.scatter(*X[core].T, s=11, c="#424242", zorder=3)
    ax.scatter(*X[~core].T, s=5, c=NOISE, zorder=2)
    for r in range(n_ranks):
        ax.text(
            X[rank == r, 0].mean(),
            X[:, 1].max() + 0.12,
            f"rank {r}",
            ha="center",
            fontsize=7,
            color="#555555",
        )

    ax = axes[1]
    panel(ax, f"(b) local MSTs: {len(subtree_ids)} subtrees")
    for i, j in local:
        ax.plot(
            *X[[i, j]].T, color=colour_of[int(subtree[i])], lw=1.1, alpha=0.7, zorder=1
        )
    draw_edges(ax, X, cut, colour="#c0392b", width=0.7, alpha=0.6)
    ax.scatter(
        *X[core_ids].T, s=12, c=[colour_of[int(subtree[i])] for i in core_ids], zorder=3
    )

    ax = axes[2]
    panel(ax, "(c) subtrees as super-vertices")
    centre = {
        s: X[[i for i in core_ids if subtree[i] == s]].mean(axis=0) for s in subtree_ids
    }
    size = {s: sum(1 for i in core_ids if subtree[i] == s) for s in subtree_ids}
    links = {tuple(sorted((int(subtree[i]), int(subtree[j])))) for i, j in cut}
    for a, b in links:
        ax.plot(*np.array([centre[a], centre[b]]).T, color="#c0392b", lw=1.2, zorder=1)
    for s in subtree_ids:
        ax.scatter(
            *centre[s],
            s=18 + 6 * size[s],
            c=colour_of[s],
            edgecolors="#333333",
            lw=0.5,
            zorder=3,
        )

    ax = axes[3]
    panel(ax, "(d) cut MST: the clusters")
    ax.scatter(*X.T, s=13, c=cluster_colours(final), zorder=3)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "distributed.svg", bbox_inches="tight", metadata={"Date": None})
    print(
        f"distributed: {len(subtree_ids)} subtrees, {len(cut)} cut edges, {len(links)} subtree links"
    )


def figure_eps_boundary() -> None:
    side = 6
    X = np.array([(x, y) for y in range(side) for x in range(side)], dtype=float)
    distances, neighbors = NearestNeighbors(n_neighbors=4).fit(X).kneighbors()
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9))
    for ax, strict in zip(axes, [False, True], strict=True):
        eps = np.nextafter(np.float32(1), np.float32(0)) if strict else 1.0
        labels = knn_dbscan_from_graph(neighbors, distances, float(eps), 5)
        core = distances[:, 3] <= np.float32(eps)
        panel(
            ax,
            r"(b) $d < \varepsilon$: no core points"
            if strict
            else r"(a) $d \leq \varepsilon$: core in the interior",
        )
        centre = 2 * side + 2
        ax.add_patch(
            Circle(X[centre], 1.0, fill=False, ls="--", lw=0.8, color="#1f5fa8")
        )
        kinds = np.where(core, 0, np.where(labels >= 0, 1, 2))
        for kind, (colour, marker, label) in enumerate(
            [
                ("#1f5fa8", "o", "core"),
                ("#e1a800", "o", "border"),
                ("#757575", "x", "noise"),
            ]
        ):
            ax.scatter(
                *X[kinds == kind].T,
                s=22,
                c=colour,
                marker=marker,
                label=label,
                zorder=3,
            )
        ax.set_xlim(-0.6, side - 0.4)
        ax.set_ylim(-0.6, side - 0.4)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(1.1, -0.04),
        ncol=3,
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "eps_boundary.svg", bbox_inches="tight", metadata={"Date": None})


def figure_sentinel(graph: Graph, ours: np.ndarray) -> None:
    X, core = graph.X, graph.core
    edges = graph.core_edges()
    weights = np.array([np.linalg.norm(X[i] - X[j]) for i, j in edges])
    sentinel = float(np.quantile(weights, 0.6))
    kept = [e for e, w in zip(edges, weights, strict=True) if w < sentinel]
    fragments = components(len(X), kept)
    fragment_ids = sorted({int(fragments[i]) for i in np.flatnonzero(core)})
    colour_of = {s: PALETTE[k % len(PALETTE)] for k, s in enumerate(fragment_ids)}
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9))
    panel(axes[0], "(a) infinite sentinel: 3 clusters")
    draw_edges(axes[0], X, edges, colour="#8a8a8a", width=0.6)
    axes[0].scatter(*X.T, s=12, c=cluster_colours(ours), zorder=3)
    panel(axes[1], f"(b) finite sentinel: {len(fragment_ids)} pieces")
    draw_edges(axes[1], X, kept, colour="#8a8a8a", width=0.6)
    dropped = [e for e, w in zip(edges, weights, strict=True) if w >= sentinel]
    draw_edges(axes[1], X, dropped, colour="#c0392b", width=0.5, alpha=0.35)
    axes[1].scatter(
        *X[core].T,
        s=12,
        c=[colour_of[int(fragments[i])] for i in np.flatnonzero(core)],
        zorder=3,
    )
    axes[1].scatter(*X[~core].T, s=5, c=NOISE, zorder=2)
    for ax in axes:
        limits(ax, X)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "sentinel.svg", bbox_inches="tight", metadata={"Date": None})
    print(f"sentinel: {len(fragment_ids)} pieces")


def main() -> None:
    graph, ours, theirs = choose_dataset()
    figure_pipeline(graph, ours, theirs)
    figure_boruvka(graph, ours)
    figure_distributed(graph, ours)
    figure_eps_boundary()
    figure_sentinel(graph, ours)


if __name__ == "__main__":
    main()
