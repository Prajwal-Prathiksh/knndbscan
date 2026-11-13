"""Draw the panels of the report's storyboard appendix (docs/report/figures/*.svg).

Usage (from the repository root): uv run python docs/report/figures/make_figures.py

Every panel is computed, not drawn by hand: the kNN graph and point classes follow the
paper's definitions, Borůvka's rounds are replayed exactly as the report's Algorithm 1
states them, and the final clusterings come from the knndbscan library and from
scikit-learn's DBSCAN. A replay that disagrees with the library stops the script. The
numbers the report quotes about the example are written to example.json, which the
report reads, so text and panels cannot drift apart.
"""

import json
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
CORE, BORDER, DARK, RED = "#1f5fa8", "#e1a800", "#424242", "#c0392b"
GREY, LIGHT, NOISE = "#8a8a8a", "#d6d6d6", "#bdbdbd"
RANK_TINTS = ["#e3ecf7", "#fbeedd", "#e6f2e3"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "path",
    # A fixed salt and no date make the SVGs identical from run to run.
    "svg.hashsalt": "knndbscan",
    "axes.linewidth": 0.6, "axes.edgecolor": "#777777",
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


def choose_dataset() -> tuple[int, Graph, np.ndarray, np.ndarray]:
    """The first seed whose data shows a DBSCAN merge that kNN-DBSCAN keeps apart."""
    for seed in range(500):
        graph = build(make_data(seed))
        ours = knn_dbscan_from_graph(graph.neighbors, graph.distances, EPS, MIN_SAMPLES)
        theirs = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(graph.X)
        border = (~graph.core & (ours >= 0)).sum()
        if (
            ours.max() == 2
            and theirs.max() == 1
            and border >= 4
            and (ours == -1).sum() >= 5
        ):
            return seed, graph, ours, theirs
    raise SystemExit("no seed shows the DBSCAN/kNN-DBSCAN difference")


def boruvka_rounds(graph: Graph) -> list[dict]:
    """Replay Algorithm 1 (local Borůvka); record each round's labels, selections and forest.

    A round's entry holds the subtrees at its start, the edges it selects (with whether the
    two subtrees chose each other) and the forest of all edges selected before it.
    """
    core = graph.core
    R = {int(i): int(i) for i in np.flatnonzero(core)}
    L = dict.fromkeys(R, 0)
    forest: list[tuple[int, int]] = []
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
        mutual = {u for u in best if p[p[u]] == u}
        for u in subtrees:  # break symmetry between mutual pairs
            if p[p[u]] == u and u < p[u]:
                p[u] = u
        while True:  # pointer jumping
            jumped = {u: p[p[u]] for u in subtrees}
            if jumped == p:
                break
            p = jumped
        selected = [(i, q, u in mutual) for u, (_, i, q) in best.items()]
        rounds.append({"labels": dict(R), "selected": selected, "forest": list(forest)})
        forest += [(i, q) for i, q, _ in selected]
        R = {i: p[R[i]] for i in R}
        if len(set(R.values())) == len(subtrees):
            rounds.append({"labels": dict(R), "selected": [], "forest": list(forest)})
            return rounds


def components(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    rows, cols = (np.array([e[k] for e in edges], dtype=int) for k in (0, 1))
    graph = coo_matrix((np.ones(len(edges)), (rows, cols)), shape=(n, n))
    return connected_components(graph, directed=True, connection="weak")[1]


def by_first_member(labels: dict[int, int]) -> dict[int, int]:
    """Number subtrees 0, 1, ... in order of their lowest point, as the library numbers clusters."""
    roots = sorted(
        set(labels.values()), key=lambda r: min(i for i, s in labels.items() if s == r)
    )
    return {r: k for k, r in enumerate(roots)}


def cluster_colours(labels: np.ndarray) -> list[str]:
    return [PALETTE[label % len(PALETTE)] if label >= 0 else NOISE for label in labels]


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


def save(name: str, draw, X: np.ndarray, size=(3.0, 3.0), background=None) -> None:
    """Draw one storyboard panel: square, no ticks, the bounds of X plus a margin."""
    fig, ax = plt.subplots(figsize=size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3)
    ax.set_ylim(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3)
    if background:
        background(ax)
    draw(ax)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def legend(ax, loc="upper left") -> None:
    ax.legend(
        loc=loc, fontsize=6.5, frameon=False, handletextpad=0.2, borderaxespad=0.3
    )


def story_definitions(graph: Graph, ours: np.ndarray, theirs: np.ndarray) -> dict:
    """Panels for Algorithm 2 on one rank: graph, core points, components, labels, DBSCAN."""
    X, core = graph.X, graph.core
    border = ~core & (ours >= 0)
    noise = ours == -1
    directed = [
        (i, int(j))
        for i in range(len(X))
        for j in graph.neighbors[i, : MIN_SAMPLES - 1]
    ]

    def knn_graph(ax):
        draw_edges(
            ax, X, directed, colour="#a8a8a8", width=0.45, arrows=True, alpha=0.85
        )
        ax.scatter(*X.T, s=9, c=DARK, zorder=3)

    # One core and one non-core point, with their eps-disks: the disk of a core point holds
    # at least M points (itself included), the other's does not.
    a_core, a_noise = np.flatnonzero(core)[5], np.flatnonzero(noise)[0]

    def core_points(ax):
        for i, colour in [(a_core, CORE), (a_noise, "#757575")]:
            ax.add_patch(
                Circle(X[i], EPS, fill=False, ls="--", lw=0.7, color=colour, zorder=2)
            )
        ax.scatter(*X[core].T, s=12, c=CORE, zorder=3, label="core")
        ax.scatter(*X[~core].T, s=10, c=NOISE, zorder=3, label="not core")
        legend(ax)

    edges = graph.core_edges()

    def core_components(ax):
        draw_edges(ax, X, edges, colour=GREY, width=0.7)
        ax.scatter(*X[~core].T, s=6, c=LIGHT, zorder=2)
        idx = np.flatnonzero(core)
        ax.scatter(*X[idx].T, s=13, c=[cluster_colours(ours)[i] for i in idx], zorder=3)

    # A border point, joined to the core neighbor whose cluster it takes.
    b = np.flatnonzero(border)[0]
    q = next(int(j) for j in graph.neighbors[b] if core[j])

    def labels(ax):
        colours = cluster_colours(ours)
        ax.add_patch(
            Circle(X[b], EPS, fill=False, ls="--", lw=0.7, color=BORDER, zorder=2)
        )
        ax.plot(*X[[b, q]].T, color=DARK, lw=1.0, zorder=2)
        idx = np.flatnonzero(core)
        ax.scatter(*X[idx].T, s=12, c=[colours[i] for i in idx], zorder=3)
        idx = np.flatnonzero(border)
        ax.scatter(*X[idx].T, s=22, c=[colours[i] for i in idx], edgecolors=DARK, lw=0.8,
                   zorder=4, label="border")  # fmt: skip
        ax.scatter(
            *X[noise].T, s=16, marker="x", c="#757575", zorder=3, label="noise (−1)"
        )
        legend(ax)

    # Give each DBSCAN cluster the colour of the kNN-DBSCAN cluster it mostly contains.
    matched = np.full_like(theirs, -1)
    for cluster in np.unique(theirs[theirs >= 0]):
        matched[theirs == cluster] = np.bincount(
            ours[(theirs == cluster) & (ours >= 0)]
        ).argmax()
    bridges = [(i, j) for i in np.flatnonzero(core) for j in np.flatnonzero(core)
               if i < j and ours[i] != ours[j] and np.linalg.norm(X[i] - X[j]) <= EPS]  # fmt: skip

    def dbscan(ax):
        ax.scatter(*X.T, s=13, c=cluster_colours(matched), zorder=3)
        draw_edges(ax, X, bridges, colour=RED, width=1.5, zorder=2)

    for name, draw in [("knn-graph", knn_graph), ("core-points", core_points),
                       ("components", core_components), ("labels", labels), ("dbscan", dbscan)]:  # fmt: skip
        save(name, draw, X)
    return {
        "n_points": len(X), "n_core": int(core.sum()), "n_border": int(border.sum()),
        "n_noise": int(noise.sum()), "n_clusters": int(ours.max() + 1),
        "n_dbscan_clusters": int(theirs.max() + 1), "n_bridges": len(bridges),
    }  # fmt: skip


def story_boruvka(graph: Graph, ours: np.ndarray) -> dict:
    """Panels for Algorithm 1: the selections of each round on the subtrees they start from."""
    X, core = graph.X, graph.core
    rounds = boruvka_rounds(graph)
    final = rounds[-1]["labels"]
    ids = by_first_member(final)
    replay = np.array([ids[final[i]] if i in final else -1 for i in range(len(X))])
    assert np.array_equal(replay[core], ours[core]), (
        "the Borůvka replay disagrees with the library"
    )
    edges = graph.core_edges()

    def state(k: int, arrows: bool):
        labels, forest = rounds[k]["labels"], rounds[k]["forest"]
        # Single points are dark, so that the first round is not a confetti of colours.
        sizes = {r: list(labels.values()).count(r) for r in set(labels.values())}
        order = by_first_member(labels)
        colour = {
            r: (DARK if sizes[r] == 1 else PALETTE[order[r] % len(PALETTE)])
            for r in sizes
        }

        def draw(ax):
            draw_edges(ax, X, edges, colour="#ececec", width=0.5)
            for i, j in forest:
                ax.plot(
                    *X[[i, j]].T, color=colour[labels[i]], lw=1.4, alpha=0.75, zorder=2
                )
            if arrows:
                for i, j, mutual in rounds[k]["selected"]:
                    ax.add_patch(FancyArrowPatch(X[i], X[j], arrowstyle="-|>", mutation_scale=8,
                                                 color=RED if mutual else "#212121", lw=0.9,
                                                 shrinkA=2.5, shrinkB=2.5, zorder=4))  # fmt: skip
            idx = np.array(sorted(labels))
            ax.scatter(*X[idx].T, s=15, c=[colour[labels[i]] for i in idx], edgecolors="white",
                       lw=0.3, zorder=3)  # fmt: skip
            ax.scatter(*X[~core].T, s=5, c=LIGHT, zorder=1)

        return draw

    for k in range(3):
        save(f"boruvka-{k + 1}", state(k, arrows=True), X)
    save("boruvka-merged-1", state(1, arrows=False), X)
    save("boruvka-end", state(len(rounds) - 1, arrows=False), X)
    counts = [len(set(r["labels"].values())) for r in rounds]
    return {
        "rounds": counts[
            :-1
        ],  # subtrees at the start of each round that merged something
        "mutual": [sum(m for *_, m in r["selected"]) // 2 for r in rounds[:3]],
    }


def story_distributed(graph: Graph, ours: np.ndarray, n_ranks: int = 3) -> dict:
    """Panels for Algorithm 2 on several ranks: blocks, local subtrees, super-vertices, clusters."""
    # Number the points left to right, so that each rank's contiguous block is a vertical strip.
    order = np.argsort(graph.X[:, 0])
    position = np.argsort(order)
    X = graph.X[order]
    neighbors = position[graph.neighbors[order]]
    core = graph.core[order]
    final = ours[order]
    rank = np.arange(len(X)) // -(-len(X) // n_ranks)
    edges = [(i, int(j)) for i in np.flatnonzero(core)
             for j in neighbors[i, : MIN_SAMPLES - 1] if core[j]]  # fmt: skip
    local = [(i, j) for i, j in edges if rank[i] == rank[j]]
    cut = [(i, j) for i, j in edges if rank[i] != rank[j]]
    subtree = components(
        len(X), local
    )  # stage II: local components are the local subtrees
    core_ids = np.flatnonzero(core)
    subtree_ids = sorted({int(subtree[i]) for i in core_ids})
    colour_of = {s: PALETTE[k % len(PALETTE)] for k, s in enumerate(subtree_ids)}

    # Theorem 3.6: the components of the subtrees joined by cut edges are the clusters.
    index = {s: k for k, s in enumerate(subtree_ids)}
    contracted = components(len(subtree_ids), [(index[int(subtree[i])], index[int(subtree[j])])
                                                for i, j in cut])  # fmt: skip
    merged = {int(i): int(contracted[index[int(subtree[i])]]) for i in core_ids}
    pairs = {(merged[i], int(final[i])) for i in core_ids}
    assert len(pairs) == len(set(final[core])) == len(set(merged.values())), (
        "Theorem 3.6 failed"
    )

    lows = [X[rank == r, 0].min() for r in range(n_ranks)]
    highs = [X[rank == r, 0].max() for r in range(n_ranks)]

    def strips(ax):
        for r in range(n_ranks):
            lo = X[:, 0].min() - 0.3 if r == 0 else (highs[r - 1] + lows[r]) / 2
            hi = (
                X[:, 0].max() + 0.3
                if r == n_ranks - 1
                else (highs[r] + lows[r + 1]) / 2
            )
            ax.axvspan(lo, hi, color=RANK_TINTS[r], zorder=0, lw=0)
            # Above the frame, where no point can cover it.
            ax.text((lo + hi) / 2, 1.015, f"rank {r}", ha="center", va="bottom", fontsize=7,
                    color="#555555", transform=ax.get_xaxis_transform())  # fmt: skip

    def blocks(ax):
        draw_edges(ax, X, local, colour=GREY, width=0.6)
        draw_edges(ax, X, cut, colour=RED, width=1.1)
        ax.scatter(*X[core].T, s=11, c=DARK, zorder=3)
        ax.scatter(*X[~core].T, s=5, c=NOISE, zorder=2)

    def local_subtrees(ax):
        for i, j in local:
            ax.plot(
                *X[[i, j]].T,
                color=colour_of[int(subtree[i])],
                lw=1.2,
                alpha=0.75,
                zorder=1,
            )
        draw_edges(ax, X, cut, colour=RED, width=0.7, alpha=0.5)
        ax.scatter(
            *X[core_ids].T,
            s=12,
            c=[colour_of[int(subtree[i])] for i in core_ids],
            zorder=3,
        )

    centre = {
        s: X[[i for i in core_ids if subtree[i] == s]].mean(axis=0) for s in subtree_ids
    }
    size = {s: int(sum(subtree[i] == s for i in core_ids)) for s in subtree_ids}
    links = {tuple(sorted((int(subtree[i]), int(subtree[j])))) for i, j in cut}

    def super_vertices(ax):
        for a, b in links:
            ax.plot(*np.array([centre[a], centre[b]]).T, color=RED, lw=1.3, zorder=1)
        for s in subtree_ids:
            ax.scatter(*centre[s], s=18 + 6 * size[s], c=colour_of[s], edgecolors="#333333",
                       lw=0.5, zorder=3)  # fmt: skip

    def clusters(ax):
        ax.scatter(*X.T, s=13, c=cluster_colours(final), zorder=3)

    for name, draw in [("ranks", blocks), ("local-subtrees", local_subtrees),
                       ("super-vertices", super_vertices), ("rank-clusters", clusters)]:  # fmt: skip
        save(name, draw, X, background=strips)
    return {"n_ranks": n_ranks, "n_local_subtrees": len(subtree_ids), "n_cut_edges": len(cut),
            "n_links": len(links)}  # fmt: skip


def panel_pair(name: str, draw_left, draw_right, titles, X, size=(3.3, 1.8)) -> None:
    """Two panels side by side, for a correct and a defective variant of one step."""
    fig, axes = plt.subplots(1, 2, figsize=size)
    for ax, draw, title in zip(axes, (draw_left, draw_right), titles, strict=True):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8.5, loc="left", pad=3)
        draw(ax)
        ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    fig.tight_layout(pad=0.2, w_pad=0.8)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def eps_boundary() -> dict:
    """A unit grid, where every interior point has four neighbors at exactly eps = 1."""
    side, m = 6, 5
    X = np.array([(x, y) for y in range(side) for x in range(side)], dtype=float)
    distances, neighbors = NearestNeighbors(n_neighbors=m - 1).fit(X).kneighbors()
    distances = distances.astype(np.float32)

    def variant(strict: bool):
        eps = float(np.nextafter(np.float32(1), np.float32(0))) if strict else 1.0
        labels = knn_dbscan_from_graph(neighbors, distances, eps, m)
        kinds = np.where(
            distances[:, m - 2] <= np.float32(eps), 0, np.where(labels >= 0, 1, 2)
        )

        def draw(ax):
            ax.add_patch(
                Circle(X[2 * side + 2], 1.0, fill=False, ls="--", lw=0.8, color=CORE)
            )
            for kind, (colour, marker) in enumerate(
                [(CORE, "o"), (BORDER, "o"), ("#757575", "x")]
            ):
                ax.scatter(*X[kinds == kind].T, s=16, c=colour, marker=marker, zorder=3)

        return draw

    panel_pair("eps-boundary", variant(False), variant(True),
               [r"$d \leq \varepsilon$", r"$d < \varepsilon$"], X)  # fmt: skip
    return {"side": side, "min_samples": m}


def sentinel(graph: Graph, ours: np.ndarray) -> dict:
    """Clusters when the no-edge weight is finite: edges at or above it are never selected."""
    X, core = graph.X, graph.core
    edges = graph.core_edges()
    weights = np.array([np.linalg.norm(X[i] - X[j]) for i, j in edges])
    cutoff = float(np.quantile(weights, 0.6))
    kept = [e for e, w in zip(edges, weights, strict=True) if w < cutoff]
    dropped = [e for e, w in zip(edges, weights, strict=True) if w >= cutoff]
    pieces = components(len(X), kept)
    piece_ids = sorted({int(pieces[i]) for i in np.flatnonzero(core)})
    colour_of = {s: PALETTE[k % len(PALETTE)] for k, s in enumerate(piece_ids)}

    def infinite(ax):
        draw_edges(ax, X, edges, colour=GREY, width=0.5)
        ax.scatter(*X.T, s=7, c=cluster_colours(ours), zorder=3)

    def finite(ax):
        draw_edges(ax, X, kept, colour=GREY, width=0.5)
        draw_edges(ax, X, dropped, colour=RED, width=0.5, alpha=0.35)
        idx = np.flatnonzero(core)
        ax.scatter(*X[idx].T, s=7, c=[colour_of[int(pieces[i])] for i in idx], zorder=3)
        ax.scatter(*X[~core].T, s=4, c=NOISE, zorder=2)

    panel_pair("sentinel", infinite, finite, [r"$s = +\infty$", r"finite $s$"], X)
    return {"sentinel_pieces": len(piece_ids)}


def zoom(ax, centre: np.ndarray, half: float, title: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    if title:
        ax.set_title(title, fontsize=8.5, loc="left", pad=3)


def save_figure(fig, name: str) -> None:
    fig.tight_layout(pad=0.2, w_pad=0.8)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def story_intuition(graph: Graph, ours: np.ndarray) -> dict:
    """The opening picture, and zooms that show why Lemma 3.1 holds and where DBSCAN differs."""
    X, core = graph.X, graph.core
    save("points", lambda ax: ax.scatter(*X.T, s=10, c=DARK, zorder=3), X)

    # Lemma 3.1 seen on one point: its eps-disk holds M points exactly when its
    # (M-1)-th neighbor lies inside the disk.
    kth = graph.distances[:, MIN_SAMPLES - 2]
    near = np.flatnonzero(
        core & (kth > 0.75 * EPS)
    )  # a core point with its last neighbor near the rim
    border = np.flatnonzero(~core & (ours >= 0))
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4))
    for ax, i, title in [
        (axes[0], int(near[0]), "core"),
        (axes[1], int(border[0]), "not core"),
    ]:
        zoom(ax, X[i], 0.85, title)
        ax.scatter(*X.T, s=8, c=LIGHT, zorder=1)
        colour = CORE if core[i] else BORDER
        ax.add_patch(
            Circle(X[i], EPS, fill=False, ls="--", lw=0.8, color=colour, zorder=2)
        )
        for rank, j in enumerate(graph.neighbors[i, : MIN_SAMPLES - 1], start=1):
            last = rank == MIN_SAMPLES - 1
            ax.plot(
                *X[[i, j]].T,
                color=RED if last else GREY,
                lw=1.1 if last else 0.7,
                zorder=2,
            )
            ax.scatter(*X[j], s=14, c=DARK, zorder=3)
            ax.annotate(str(rank), X[j], xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color=RED if last else "#555555")  # fmt: skip
        ax.scatter(*X[i], s=34, c=colour, edgecolors=DARK, lw=0.6, zorder=4)
    save_figure(fig, "zoom-core")

    # A DBSCAN bridge: two core points within eps of each other, in different kNN-DBSCAN
    # clusters, because neither is among the other's M-1 nearest neighbors.
    bridges = [(int(i), int(j)) for i in np.flatnonzero(core) for j in np.flatnonzero(core)
               if i < j and ours[i] != ours[j] and np.linalg.norm(X[i] - X[j]) <= EPS]  # fmt: skip
    i, j = bridges[0]
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    zoom(ax, (X[i] + X[j]) / 2, 0.75)
    colours = cluster_colours(ours)
    ax.scatter(*X.T, s=9, c=[colours[t] for t in range(len(X))], alpha=0.35, zorder=1)
    for a in (i, j):
        for b in graph.neighbors[a, : MIN_SAMPLES - 1]:
            ax.add_patch(FancyArrowPatch(X[a], X[b], arrowstyle="-|>", mutation_scale=8,
                                         color=colours[a], lw=0.9, shrinkA=3, shrinkB=3,
                                         zorder=3))  # fmt: skip
    ax.plot(*X[[i, j]].T, color=RED, lw=1.3, ls="--", zorder=2)
    ax.annotate(r"$d \leq \varepsilon$", (X[i] + X[j]) / 2, xytext=(0, -12),
                textcoords="offset points", ha="center", fontsize=8, color=RED)  # fmt: skip
    ax.scatter(
        *X[[i, j]].T,
        s=40,
        c=[colours[i], colours[j]],
        edgecolors=DARK,
        lw=0.7,
        zorder=4,
    )
    save_figure(fig, "bridge")

    # The k-distance curve that eps is chosen from: points left of the crossing are core.
    order = np.sort(kth)
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    ax.scatter(
        np.arange(len(order)),
        order,
        s=7,
        c=np.where(order <= EPS, CORE, NOISE),
        zorder=3,
    )
    ax.axhline(EPS, color=CORE, ls="--", lw=0.8)
    ax.text(1, EPS + 0.03, r"$\varepsilon$", color=CORE, fontsize=9)
    ax.set_xlabel("points, by distance to their $(M-1)$-th neighbor", fontsize=7.5)
    ax.set_ylabel("distance", fontsize=7.5)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save_figure(fig, "kdist")
    return {
        "bridge_rank_of_partner": int(
            np.searchsorted(graph.distances[i], np.linalg.norm(X[i] - X[j])) + 1
        )
    }


def functional_cycles(target: np.ndarray) -> list[list[int]]:
    """The cycles of the map i -> target[i] (each vertex's selected edge)."""
    state = np.zeros(len(target), dtype=int)  # 0 unseen, 1 on the current path, 2 done
    cycles = []
    for start in range(len(target)):
        path, v = [], start
        while state[v] == 0:
            state[v] = 1
            path.append(v)
            v = int(target[v])
        if state[v] == 1:
            cycles.append(path[path.index(v) :])
        for u in path:
            state[u] = 2
    return cycles


def selections(graph: Graph) -> dict:
    """First-round selections on real data and on a random directed graph, with their cycles.

    On data, the lightest edges form mutual pairs; on a random directed graph, whose
    weights obey no geometry, they also form longer cycles: the cases that break naive
    pointer jumping, and why the tests use random graphs.
    """
    X, core = graph.X, graph.core
    idx = np.flatnonzero(core)
    position = {int(i): k for k, i in enumerate(idx)}
    first = np.array(
        [position[next(int(j) for j in graph.neighbors[i] if core[j])] for i in idx]
    )
    data_cycles = functional_cycles(first)

    n = 36
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        Y = rng.uniform(0, 1, (n, 2))
        target = np.array(
            [rng.choice([j for j in range(n) if j != i]) for i in range(n)]
        )
        cycles = functional_cycles(target)
        lengths = sorted(len(c) for c in cycles)
        if 4 in lengths and 3 in lengths:
            break

    def draw(ax, P, target, cycles):
        on_cycle = {v: len(c) for c in cycles for v in c}
        for i, j in enumerate(target):
            length = (
                on_cycle.get(i) if on_cycle.get(int(j)) == on_cycle.get(i) else None
            )
            colour = RED if length and length > 2 else (DARK if length == 2 else LIGHT)
            ax.add_patch(FancyArrowPatch(P[i], P[j], arrowstyle="-|>", mutation_scale=7,
                                         color=colour, lw=1.0 if length else 0.6,
                                         shrinkA=2.5, shrinkB=2.5, zorder=3 if length else 2))  # fmt: skip
        ax.scatter(*P.T, s=10, c=DARK, zorder=4)

    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4))
    for ax, P, t, c, title in [(axes[0], X[idx], first, data_cycles, "data"),
                               (axes[1], Y, target, cycles, "random directed graph")]:  # fmt: skip
        pad = 0.08 * np.ptp(P, axis=0).max()
        zoom(
            ax,
            (P.min(axis=0) + P.max(axis=0)) / 2,
            np.ptp(P, axis=0).max() / 2 + pad,
            title,
        )
        draw(ax, P, t, c)
    save_figure(fig, "selections")
    longer = sorted(len(c) for c in cycles if len(c) > 2)
    return {"data_pairs": sum(len(c) == 2 for c in data_cycles),
            "data_longer_cycles": sum(len(c) > 2 for c in data_cycles),
            "random_pairs": sum(len(c) == 2 for c in cycles), "random_longer_cycles": longer}  # fmt: skip


def performance() -> None:
    """The paper's measurements (Chen, Ruys and Biros, Sec. 5) as bars."""
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(5.4, 1.7), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    a.barh([1, 0], [400, 1], color=[GREY, CORE], height=0.55)
    a.set_yticks([1, 0], ["kNN graph", "clustering"], fontsize=7.5)
    a.set_xlabel("time, relative to clustering (64M points, 5-D, k = 100)", fontsize=7)
    a.text(400, 1, " 400×", va="center", fontsize=7.5)
    a.text(1, 0, "  1×", va="center", fontsize=7.5, color=CORE)
    a.set_xlim(0, 470)
    dims, speedup = ["3-D", "5-D", "32-D"], [13.8, 16.8, 22.2]
    b.bar(dims, speedup, color=CORE, width=0.55)
    b.axhline(64, color=GREY, ls="--", lw=0.8)
    b.text(2.3, 64, "ideal 64", va="bottom", ha="right", fontsize=7, color="#555555")
    for k, v in enumerate(speedup):
        b.text(k, v + 1.5, f"{v}", ha="center", fontsize=7)
    b.set_ylabel("speedup, 4 → 256 ranks", fontsize=7)
    b.set_ylim(0, 72)
    for ax in (a, b):
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    save_figure(fig, "performance")


def main() -> None:
    seed, graph, ours, theirs = choose_dataset()
    numbers = {"seed": seed, "eps": EPS, "min_samples": MIN_SAMPLES}
    # The report draws its own diagrams and colour keys in the panels' colours.
    numbers["colours"] = {"core": CORE, "border": BORDER, "noise": "#757575", "cut": RED,
                          "dark": DARK, "grey": GREY, "light": LIGHT, "palette": PALETTE}  # fmt: skip
    numbers |= story_intuition(graph, ours)
    numbers |= story_definitions(graph, ours, theirs)
    numbers |= story_boruvka(graph, ours)
    numbers |= story_distributed(graph, ours)
    numbers |= {"eps_grid": eps_boundary()}
    numbers |= sentinel(graph, ours)
    numbers |= selections(graph)
    performance()
    (OUT / "example.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps(numbers))


if __name__ == "__main__":
    main()
