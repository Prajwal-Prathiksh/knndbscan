#import "companion/lib.typ": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

#show: report.with(
  title: [kNN-DBSCAN: algorithm, distributed implementation and correctness],
  subtitle: [A technical companion to Chen, Ruys and Biros, "KNN-DBSCAN: a DBSCAN in high dimensions"],
  authors: ((name: [knndbscan project], affiliation: [github.com/Prajwal-Prathiksh/knndbscan]),),
  date: [November 2026 · knndbscan 0.3],
  abstract: [
    kNN-DBSCAN replaces DBSCAN's range queries by a $k$-nearest-neighbor graph, which can
    be computed approximately and at scale in high dimensions. This companion explains the
    method of Chen, Ruys and Biros @paper to readers who want to understand, use or modify
    an implementation without first reading the paper. It follows one small example from
    start to end: why the graph classifies points exactly as DBSCAN does, why its clusters
    refine DBSCAN's, how Borůvka's algorithm grows them as trees, what changes on many
    machines, and which details an implementation must get right. Every idea is shown as a
    picture, explained in words and stated formally, and every claim about the
    implementation is checked by the project's tests.
  ],
)

// ------------------------------------------------------------------- shared vocabulary
// Numbers and colours of the running example, written by figures/make_figures.py with its
// panels, so that the text, the panels and the diagrams below cannot disagree.
#let ex = json("figures/example.json")
#let core-colour = rgb(ex.colours.core)
#let border-colour = rgb(ex.colours.border)
#let noise-colour = rgb(ex.colours.noise)
#let cut-colour = rgb(ex.colours.cut)
#let panel(name, width: 100%) = image("figures/" + name + ".svg", width: width)

// Pointer states as a small graph: `pointers` maps each node to its target, and a node
// pointing at itself looks like a root, so it is shaded and drawn with a loop.
#let pointer-state(positions, pointers, bend: 14deg, spacing: 2.3em) = {
  let roots = pointers.filter(((u, v)) => u == v).map(((u, v)) => u)
  diagram(
    spacing: (spacing, spacing),
    node-stroke: 0.6pt,
    node-shape: circle,
    node-inset: 3pt,
    ..positions.pairs().map(((name, pos)) => node(
      pos, text(9pt)[$#name$], name: label(name), fill: if name in roots { rgb("#f6d7d2") } else { white },
    )),
    ..pointers.map(((u, v)) => if u == v {
      edge(label(u), label(u), "-|>", bend: 130deg, stroke: 0.6pt + cut-colour)
    } else {
      edge(label(u), label(v), "-|>", bend: bend, stroke: 0.6pt)
    }),
  )
}
#let square = (a: (0, 0), b: (1, 0), c: (1, 1), d: (0, 1))
#let triangle = (a: (0, 0), b: (1, 0), c: (0.5, 0.9))
#let pair = (u: (0, 0), v: (1, 0))
#let chain = (a: (0, 0), b: (1, 0), c: (2, 0), d: (3, 0), e: (4, 0), f: (5, 0), g: (6, 0), h: (7, 0))
// Each node of the chain a <- b <- ... <- h after t rounds of pointer jumping: 2^t steps back.
#let jumped(t) = {
  let names = chain.keys()
  names.enumerate().map(((k, name)) => (name, names.at(calc.max(0, k - calc.pow(2, t)))))
}
#let small(body) = text(8pt, body)

// ---------------------------------------------------------------------- the algorithm
#let driver = (
  require[this rank's rows of the kNN graph, sorted, without the point itself; $epsilon$; $M$],
  ensure[a cluster label for each of this rank's points, $-1$ for noise],
  ln(0, [$C <- {i : "distance in column" M - 1 "of row" i <= epsilon}$], note: [core points (@sec:classify)]),
  ln(0, [$R$, cut edges $<-$ local Borůvka over $C$], note: [@alg:local]),
  ln(0, [number this rank's subtrees after those of lower ranks], note: [`Allgather`]),
  ln(0, [rewrite each cut edge $(i, j)$ as $(R[i], R[j])$], note: [`Alltoallv` to $j$'s rank]),
  ln(0, [roots $<-$ cut MST over the subtrees], note: [@alg:cut]),
  ln(0, [$R[i] <-$ root of $R[i]$ for every $i in C$]),
  ln(0, [*parallel for* each non-core point $i$], note: [border points]),
  ln(1, [$q <-$ first core neighbor of $i$ within $epsilon$, preferring one on this rank], note: [`Alltoallv` if remote]),
  ln(1, [$R[i] <- R[q]$ if $q$ exists, otherwise $-1$], note: [noise]),
)
#let one-rank = (label: <alg:driver>, lines: driver)

= What do you see? <sec:intro>

#question[Look at these points. Which of them belong together?]

#scene(side: left, width: 34%, panel("points"))[
  Most people see three groups and a handful of stragglers, without being told how many
  groups to look for or what shape they have. _Density-based clustering_ turns that
  perception into a rule: a group is a region where points are packed closely, and a
  straggler is a point in a sparse region. DBSCAN @ester1996 is the classic such rule. It
  needs no number of clusters, finds clusters of any shape, and sets outliers aside as
  noise.

  Its cost is the question it asks of every point: _which points lie within distance
  $epsilon$ of you?_ In two dimensions a spatial tree answers this quickly. As the
  dimension grows, trees degrade towards comparing every pair of points, $O(n^2)$, and
  every new value of $epsilon$ asks again.
]

#scene(width: 34%, panel("knn-graph"))[
  kNN-DBSCAN @paper asks a different question: _who are your $k$ nearest neighbors?_
  The answers form a $k$-nearest-neighbor graph ($k$-NNG): an arrow from each point to
  each of its $k$ nearest neighbors. Unlike range queries, this graph can be approximated
  well in high dimensions (randomized projection trees @xiao2016, as in GOFMM @yu2018),
  it stores only $O(n k)$ edges, and it is computed once: every $epsilon$ and every
  $M <= k + 1$ can be clustered from it.

  The surprise is how little is lost. The graph classifies every point exactly as DBSCAN
  does, and its clusters differ from DBSCAN's only by splitting a cluster that a thin
  bridge holds together.
]

#filmstrip(
  (panel("points"), [points]),
  (panel("knn-graph"), [kNN graph \ @sec:intro]),
  (panel("core-points"), [core points \ @sec:classify]),
  (panel("components"), [linked cores \ @sec:reach]),
  (panel("boruvka-end"), [trees \ @sec:forest]),
  (panel("labels"), [clusters, noise \ @sec:classify]),
  caption: [The whole method on the running example, one frame per idea. Each later section
    explains one frame, then states it formally.],
) <fig:pipeline>

The running example has #ex.n_points points in three blobs, with $M = #ex.min_samples$
and $epsilon = #ex.eps$. Its panels are computed, not drawn: `figures/make_figures.py`
replays the algorithm exactly as the pseudocode states it, stops unless the replay finds
the library's clusters, and writes the numbers this text quotes.

== The algorithm at a glance

All of kNN-DBSCAN fits in nine lines. They are given here once in full; whenever a later
picture is about some of them, the algorithm appears again beside the picture with those
lines highlighted, so there is never a reason to turn back.

#algorithm(caption: [kNN-DBSCAN on one rank: `knndbscan()` (paper's Algorithm 1). On a
  single process, lines 3--5 have nothing to exchange.], pseudocode(driver)) <alg:driver>

== How to read this companion

Each idea appears three times: as a *picture* of the running example, in plain *words*,
and in *symbols*, as a definition, lemma or theorem in a tinted box. The pictures and the
symbols say the same thing, so either can be used to check the other. Colours keep one
meaning throughout: #swatch(core-colour) core point, #swatch(border-colour) border point,
#swatch(noise-colour, shape: "cross") noise, #swatch(cut-colour, shape: "line") an edge
between processes, or what goes wrong; highlighted pseudocode lines are the ones the
picture beside them shows. Numbers such as "Theorem 3.4" refer to the paper, whose
statements are restated in one notation; proofs are sketches that show why a result
holds. `docs/ALGORITHM.md` maps each line of @alg:driver to its file and function.

#translate(
  [the $n$ points], $cal(I)$,
  [the distance between two points (Euclidean in the Python package)], $d(p, q)$,
  [the radius, and the minimum number of points, the point itself included], [$epsilon$, $M$ (`eps`, `min_samples`)],
  [the points within $epsilon$ of $p$, $p$ included], $cal(N)_epsilon (p) = {q in cal(I) : d(p, q) <= epsilon}$,
  [the $M$ points nearest to $p$, $p$ included], $cal(N)_M (p)$,
  [the neighbors stored for each point, the point itself excluded], $k$,
  [the core, border and noise points], $cal(I)_"core", cal(I)_"border", cal(I)_"noise"$,
  [the number of processes (MPI ranks)], $P$,
)

#takeaway[Cluster the kNN graph instead of $epsilon$-balls: the classification of
  points stays exactly DBSCAN's, and clustering becomes growing trees, which
  parallelizes.]

= Dense or sparse, one point at a time <sec:classify>

#question[When is a point in a dense place?]

#scene(panel("core-points"))[
  Draw a disk of radius $epsilon$ around the point and count the points inside, the point
  itself included. If there are at least $M$, the point is a *core point*
  #swatch(core-colour): it lies inside a cluster. The blue dashed disk holds more than
  $M = #ex.min_samples$ points; the grey one holds fewer. Of the #ex.n_points points,
  #ex.n_core are core.

  A point that is not core can still lie inside a core point's disk, at the edge of a
  cluster: it is a *border point*. A point in no core point's disk is *noise*.
]

#definition(title: [classification of points, Def. 3.1])[
  Given $epsilon$ and $M$, a point $p$ is a _core point_ if $|cal(N)_epsilon (p)| >= M$. A
  point that is not core is a _border point_ if some core point $q$ has
  $p in cal(N)_epsilon (q)$, and _noise_ otherwise.
] <def:classes>

Counting points in a disk is a range query, the expensive operation. The kNN graph
answers the same question without one.

#story()
#step([Core points from one column], code: one-rank, focus: (1,), window: 1, width: 60%, panel("zoom-core"))[
  Number a point's neighbors by distance, $1, 2, dots, M - 1$. The disk holds $M$ points,
  the point and $M - 1$ others, exactly when neighbor $M - 1$ (red) is inside it. Left,
  the 5th neighbor is inside: core. Right, it is just outside: not core. One number per
  point decides, the distance in column $M - 1$ of its row, and no communication is needed.
]

#step([Border points and noise], code: one-rank, focus: (7, 8, 9), window: 1, panel("labels"))[
  A non-core point takes the cluster of its first core neighbor within $epsilon$ (ringed;
  one is joined to the core point it follows) and is noise, $-1$, if it has none. The
  search never leaves the first $M - 1$ columns: a core point that could make it a border
  point is always among its $M - 1$ nearest neighbors. Here #ex.n_border points are
  border points and #ex.n_noise are noise.
]

#lemma(title: [Lemma 3.1])[
  + $p$ is a core point if and only if $max_(q in cal(N)_M (p)) d(p, q) <= epsilon$, that
    is, if its $(M-1)$-th nearest neighbor other than itself lies within $epsilon$.
  + $p$ is a border point if and only if $p$ is not core and some core point
    $q in cal(N)_M (p)$ has $d(p, q) <= epsilon$.
] <lem:classify>

#proof[
  (1) $cal(N)_epsilon (p)$ contains at least $M$ points exactly when the $M$-th closest
  point (counting $p$) is within $epsilon$. (2) If $p$ is not core, its $M$-th closest
  point is farther than $epsilon$, so every point within $epsilon$ of $p$, in particular a
  core point that makes $p$ a border point, is among its $M$ nearest.
]

A $k$-NNG with $k >= M - 1$ therefore classifies every point exactly as DBSCAN does,
whatever larger $k$ was stored, and border points may join any cluster whose core point
reaches them: that ambiguity is DBSCAN's own.

#takeaway[A point's class is read from its row: core if the distance in column $M - 1$
  is at most $epsilon$, border if a core point among its first $M - 1$ neighbors is within
  $epsilon$.]

= Who belongs with whom <sec:reach>

#question[Two core points are close to each other. Are they in the same cluster?]

DBSCAN says yes when they are within $epsilon$ of each other, and links clusters through
chains of such pairs. kNN-DBSCAN says yes only when one of them lists the other among its
$M - 1$ nearest neighbors. On the running example the two answers differ:

#filmstrip(
  arrows: false,
  (panel("components", width: 62%), [kNN-DBSCAN: #ex.n_clusters clusters]),
  (panel("dbscan", width: 62%), [DBSCAN: #ex.n_dbscan_clusters clusters]),
)

#scene(width: 36%, panel("bridge"))[
  Zoom into the red link. Its two core points are within $epsilon$, so DBSCAN links them
  and merges two blobs. But each point's $M - 1 = #(ex.min_samples - 1)$ nearest neighbors
  (arrows) lie in its own blob, and the other point is not among them, so kNN-DBSCAN does
  not link them and keeps the blobs apart. Every kNN link is also a DBSCAN link (a core
  point's $M - 1$ nearest neighbors are within $epsilon$), but not the other way round.
]

#translate(
  [$q$ is a step from core point $p$, for DBSCAN], [$p in cal(I)_"core"$ and $d(p, q) <= epsilon$],
  [$q$ is a step from core point $p$, for kNN-DBSCAN: either one lists the other], [$p in cal(I)_"core"$ and ($q in cal(N)_M (p)$ or $p in cal(N)_M (q)$)],
  [core points in one kNN-DBSCAN cluster], [connected in $G_(M, "core")$: core points, joined to their core neighbors among the first $M - 1$],
)

#definition(title: [reachability and DBSCAN clusters, Defs. 3.2--3.5])[
  $q$ is _directly $epsilon$-reachable_ from $p$ if $p$ is core and $q in cal(N)_epsilon (p)$;
  _$epsilon$-reachable_ if a chain of core points $p = p_0, p_1, dots, p_m$ links them by
  direct reachability, ending in $q$. A _DBSCAN cluster_ is a nonempty set $C$ that is
  closed under $epsilon$-reachability (if $p in C$ is core and $q$ is reachable from $p$,
  then $q in C$) and connected (any two members are reachable from a common core point).
]

#definition(title: [$M$-reachability and kNN-DBSCAN clusters, Defs. 3.6--3.9])[
  $q$ is _directly $M$-reachable_ from $p$ if $p$ is core and $q in cal(N)_M (p)$ or
  $p in cal(N)_M (q)$. $M$-reachability, $M$-connectedness and _kNN-DBSCAN clusters_ are
  defined from it exactly as their $epsilon$ counterparts.
] <def:mreach>

#lemma(title: [Lemmas 3.2 and 3.3])[
  On core points, $M$-reachability is an equivalence relation, and it implies
  $epsilon$-reachability.
] <lem:equivalence>

#proof[
  Symmetry comes from the symmetrized definition; reflexivity and transitivity from
  concatenating chains. If $p$ is core then every $q in cal(N)_M (p)$ satisfies
  $d(p, q) <= epsilon$ by @lem:classify, so each $M$-step is an $epsilon$-step.
]

So the core points of each kNN-DBSCAN cluster are a connected component of
$G_(M, "core")$, and its edges need no distance test: they are within $epsilon$
automatically.

#theorem(title: [relation to DBSCAN, Theorem 3.4])[
  With the same $epsilon$ and $M$:
  + every kNN-DBSCAN cluster's core points lie in a single DBSCAN cluster, so kNN-DBSCAN
    finds at least as many clusters as DBSCAN;
  + if the numbers of clusters are equal, the two clusterings of the core points coincide.
  Core, border and noise points are the same in both (@lem:classify).
] <thm:refine>

#proof[
  (1) follows from @lem:equivalence: $M$-connected core points are $epsilon$-connected.
  (2) By (1), sending each kNN-DBSCAN cluster to the DBSCAN cluster that contains its
  core points defines a map between the two sets of clusters. It is onto, because every
  core point belongs to some kNN-DBSCAN cluster. If the two sets have the same size, it
  is therefore one-to-one: no DBSCAN cluster holds two kNN-DBSCAN clusters, and the
  clusterings of the core points coincide.
]

In the paper's experiments, and in this project's tests on blobs, moons, anisotropic and
20-dimensional data, the two clusterings agree on the core points whenever their counts
agree, as part (2) guarantees.

#note(title: [Consequences for the graph])[
  Only the first $M - 1$ columns of each row are ever read: they decide core points
  (column $M - 1$), cluster edges (columns $1, dots, M - 1$) and border candidates. A
  graph stored with $k$ neighbors therefore serves every $M <= k + 1$ and every $epsilon$
  without recomputation, which makes parameter search cheap: the graph is the expensive
  part (@sec:performance).
]

#takeaway[kNN-DBSCAN links fewer pairs than DBSCAN, so its clusters refine DBSCAN's: it
  can split a cluster held together by a sparse bridge, and never merges two.]

= Clusters are trees <sec:forest>

#question[The clusters are the connected components of $G_(M, "core")$. How are the
  components of a graph with billions of edges found in parallel?]

Grow a tree through each component. A spanning forest has exactly the components of its
graph, and _Borůvka's algorithm_ grows all of its trees at once: every tree, or
_subtree_, picks its lightest edge to another subtree, and the picks merge subtrees.
Every pick merges its subtree with another, so each round at least halves the number of
subtrees that can still merge, and $O(log n)$ rounds suffice.

#filmstrip(
  (panel("boruvka-1"), [round 1: #ex.rounds.at(0) subtrees]),
  (panel("boruvka-2"), [round 2: #ex.rounds.at(1)]),
  (panel("boruvka-3"), [round 3: #ex.rounds.at(2)]),
  (panel("boruvka-end"), [#ex.rounds.at(3) trees: the clusters]),
  caption: [Borůvka on the core points of the running example. Arrows are the picks of a
    round, colours the subtrees it starts from, red arrows pairs of subtrees that picked
    each other. Each round's picks merge the subtrees into the next frame's.],
) <fig:boruvka>

#theorem(title: [Theorem 3.5])[
  The connected components of a minimum spanning forest of $G_(M, "core")$ are the
  kNN-DBSCAN clusters of the core points (and those of $G_(epsilon, "core")$ are the
  DBSCAN clusters).
] <thm:mst>

The statement is immediate, since a spanning forest has its graph's components; its value
is algorithmic. A _minimum_ spanning forest also supports hierarchical clustering; for the
clusters alone minimality is not needed (@prop:nonminimal). A round has three mechanics.

== Picking: the lightest edge is the first eligible one

#scene(width: 46%, {
  let cell(body, struck: false, picked: false) = box(
    width: 2.4em, height: 1.7em, stroke: 0.6pt,
    fill: if picked { core-colour.lighten(80%) } else { white },
    align(center + horizon, if struck { text(fill: luma(150), strike(body)) } else { body }),
  )
  grid(
    columns: 6, column-gutter: 0pt, row-gutter: 0.35em, align: center + horizon,
    small[row $i$ #h(0.4em)], cell($q_1$, struck: true), cell($q_2$, struck: true),
    cell($q_3$, picked: true), cell($q_4$), cell($q_5$),
    [], small[same \ subtree], small[not \ core], small[*picked*], [], [],
    [], [], [], text(fill: core-colour)[$arrow.t$ #small[$L[i]$]], [], [],
  )
})[
  Each row lists a point's neighbors in increasing distance, so the first neighbor that
  is a core point in another subtree is the point's lightest such edge, and the scan stops
  there. Each subtree keeps the lightest edge its points propose. An edge that has become
  internal to a subtree stays internal, so the column pointer $L[i]$ only moves forward:
  over all rounds, each of the $n (M - 1)$ edges is inspected a bounded number of times.
]

== Breaking symmetry: a mutual pair elects a root

#scene(side: left, width: 40%, grid(
  columns: 3, column-gutter: 1.2em, row-gutter: 0.5em, align: center + horizon,
  pointer-state(pair, (("u", "v"), ("v", "u")), bend: 30deg, spacing: 3em),
  text(13pt, fill: luma(100))[→],
  pointer-state(pair, (("u", "u"), ("v", "u")), spacing: 3em),
  small[$u$ and $v$ pick each other], [], small[$u < v$: $u$ is the root],
))[
  Each subtree $u$ points to the subtree $p(u)$ its edge reaches, or to itself if it has
  no edge. Two subtrees that picked each other would point at each other forever, so the
  one with the smaller id points at itself and becomes the root of both. In the running
  example every new subtree contains exactly one such pair: #ex.mutual.at(0) pairs in
  round 1 for #ex.rounds.at(1) new subtrees, #ex.mutual.at(1) in round 2 for
  #ex.rounds.at(2).
]

== Pointer jumping: everyone finds the root in $log$ steps

#figure(
  grid(
    columns: (auto, 1fr), column-gutter: 1.2em, row-gutter: 0.9em, align: (right + horizon, center + horizon),
    $p$, pointer-state(chain, jumped(0), bend: 0deg, spacing: 1.9em),
    $p^2$, pointer-state(chain, jumped(1), bend: 35deg, spacing: 1.9em),
    $p^4$, pointer-state(chain, jumped(2), bend: 35deg, spacing: 1.9em),
    $p^8$, pointer-state(chain, jumped(3), bend: 35deg, spacing: 1.9em),
  ),
  caption: [Pointer jumping on a chain of depth 7: every node replaces its pointer by its
    pointer's pointer, $p <- p compose p$, so after $t$ rounds it points $2^t$ steps
    ahead, or at the root. After $ceil(log_2 7) = 3$ rounds every node points at the
    root $a$.],
) <fig:jumping>

After pointer jumping every subtree points at the root of its new tree, and each core
point takes that root as its label. With the three mechanics, the whole local algorithm:

#let local-boruvka = (
  require[core points $C$ of this process; neighbor rows sorted by distance],
  ensure[$R[i]$, the root of the subtree that contains each $i in C$],
  ln(0, [$R[i] <- i$ and $L[i] <- 1$ for every $i in C$], note: [$L[i]$: next column of row $i$]),
  ln(0, [*repeat*]),
  ln(1, [$E_min [u] <- ("none", +infinity)$ for every subtree $u$]),
  ln(1, [*parallel for* $i in C$], note: [pick]),
  ln(2, [*while* $L[i] <= M - 1$]),
  ln(3, [$q <-$ neighbor $L[i]$ of $i$]),
  ln(3, [*if* $q$ is stored on another rank *then* record cut edge $(i, q)$; $L[i] <- L[i] + 1$]),
  ln(3, [*else if* $q in C$ *and* $R[q] != R[i]$ *then* `pwrite`$(E_min [R[i]], R[q], d(i, q))$; *break*], note: [rows are sorted]),
  ln(3, [*else* $L[i] <- L[i] + 1$], note: [non-core, or same subtree]),
  ln(1, [$p(u) <- E_min [u]."target"$, or $u$ if it has none], note: [break symmetry]),
  ln(1, [*if* $p(p(u)) = u$ *and* $u < p(u)$ *then* $p(u) <- u$], note: [mutual pair: smaller id is root]),
  ln(1, [*repeat* $p(u) <- p(p(u))$ *until* no pointer changes], note: [pointer jumping]),
  ln(1, [join the subtrees left on cycles], note: [@sec:cycles]),
  ln(1, [$R[i] <- p(R[i])$ for every $i in C$], note: [relabel]),
  ln(0, [*until* the number of subtrees does not decrease]),
)
#let local = (label: <alg:local>, lines: local-boruvka)

#algorithm(caption: [Local Borůvka over the core points of one process (paper's
  Algorithm 2). `pwrite` keeps the lighter of two proposals (paper's Algorithm 3).],
  pseudocode(local-boruvka)) <alg:local>

#story()
#step([Round 1: every subtree picks], code: local, focus: (1, 3, 4, 5, 6, 7, 8, 9), panel("boruvka-1"))[
  Every core point starts as a subtree of its own (dark). It scans its row and stops at
  the first core neighbor in another subtree, its lightest edge (arrows). Red arrows are
  the #ex.mutual.at(0) pairs that picked each other.
]

#step([Round 1: merge along the picks], code: local, focus: (10, 11, 12, 13, 14), panel("boruvka-merged-1"))[
  Each subtree points along its pick; each mutual pair elects its smaller id; pointer
  jumping takes every subtree to its root, and relabelling hands the root to every
  point. The #ex.rounds.at(0) subtrees become #ex.rounds.at(1), drawn with the edges of
  their trees.
]

#step([No edge left: the clusters], code: local, focus: (15,), panel("boruvka-end"))[
  Rounds 2 and 3 repeat lines 3--14 (@fig:boruvka). Then no subtree has an edge to
  another, the number of subtrees stays at #ex.rounds.at(3), and the loop ends. The
  coloured edges are a spanning forest of $G_(M, "core")$, one tree per cluster.
]

== When nobody is a root <sec:cycles>

#question[Symmetry breaking assumes the picks end in a mutual pair. What if they never do?]

#scene(width: 52%, panel("selections"))[
  On data they always do. With an undirected graph and distinct weights, following picks
  only ever decreases the weight, so a chain of picks can only end by returning to the
  previous subtree: a mutual pair. The example's first round has #ex.data_pairs pairs
  and #ex.data_longer_cycles longer cycles (left).

  A kNN graph is _directed_, though: $q in cal(N)_M (p)$ does not imply
  $p in cal(N)_M (q)$, so a subtree sees only the edges its own points list, and the
  weights along a chain of picks need not decrease. On a random directed graph (right)
  the picks form no pairs at all, but cycles of length
  #ex.random_longer_cycles.map(str).join(" and ") (red). Ties, frequent in duplicated or
  quantized data, create such cycles in real data too.
]

#scene(side: left, width: 30%, diagram(
  spacing: (3.2em, 2.4em),
  node-stroke: 0.6pt,
  node-shape: circle,
  node((0, 0), $V_1$, name: <v1>),
  node((1, 0), $V_2$, name: <v2>),
  node((1, 1), $V_3$, name: <v3>),
  node((0, 1), $V_4$, name: <v4>),
  edge(<v1>, <v2>, "-|>", $0.12$, label-side: left),
  edge(<v2>, <v3>, "-|>", $0.15$, label-side: left),
  edge(<v3>, <v4>, "-|>", $0.14$, label-side: left),
  edge(<v4>, <v1>, "-|>", $0.13$, label-side: left),
))[
  The paper's smallest example ($k = 2$): each vertex's lightest out-edge is drawn.
  $V_1 -> V_2$ is the lightest edge of all, so an undirected graph would make
  $V_1, V_2$ a mutual pair. But $V_2$ does not list $V_1$, and the picks close a cycle of
  four. None of its members points at itself, so none is a root, and pointer jumping has
  to cope with it.
]

What pointer jumping does on a cycle of length $ell$ depends on $ell$. After $t$ rounds
each pointer is $p^(2^t)$, which is the identity exactly when $ell$ divides $2^t$:

#filmstrip(
  (pointer-state(triangle, (("a", "b"), ("b", "c"), ("c", "a"))), [3-cycle: picks $p$]),
  (pointer-state(triangle, (("a", "c"), ("b", "a"), ("c", "b"))), [$p^2$]),
  (pointer-state(triangle, (("a", "b"), ("b", "c"), ("c", "a"))), [$p^4 = p$: rotates forever]),
)
#filmstrip(
  caption: [Pointer jumping on cycles. On an odd cycle (top) the pointers rotate and the
    number of unresolved subtrees stops falling, which is easy to detect. On a cycle whose
    length is a power of two (bottom) every member ends pointing at itself, which is
    exactly what a root looks like (shaded).],
  (pointer-state(square, (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))), [4-cycle: picks $p$]),
  (pointer-state(square, (("a", "c"), ("b", "d"), ("c", "a"), ("d", "b"))), [$p^2$]),
  (pointer-state(square, (("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"))), [$p^4$: four false "roots"]),
) <fig:cycles>

#pitfall(title: [Power-of-two cycles])[
  An implementation that detects cycles only by the count of unresolved subtrees catches
  odd cycles, but on a cycle of length 2, 4, 8, ... every member "resolves" to itself.
  Taken as roots, the members never merge, the next round picks the same edges, and
  Borůvka stops with the cluster in pieces. The fix: a subtree that was not a root and
  whose chain returns to itself is on a cycle, and stays unresolved so that cycle joining
  sees it. Sequential in-place updates hide the problem, because the first member to point
  at itself becomes the others' root; the synchronous updates of the distributed
  algorithm expose it (@sec:distributed).
]

The subtrees left unresolved form cycles with trees hanging off them. Line 13 of
@alg:local joins each such group by making one member the root of all of it, and both
ways it can go wrong produce a wrong cluster:

#story()
#step([Grouping must link roots], grid(
  columns: 2, column-gutter: 2.5em, row-gutter: 0.4em,
  diagram(
    spacing: 3.4em, node-stroke: 0.6pt, node-shape: circle, node-inset: 3pt,
    node((0, 0), text(9pt, $r_1$), name: <r1>), node((1, 0), text(9pt, $r_2$), name: <r2>),
    node((0.5, 0.9), text(9pt, $r_3$), name: <r3>),
    edge(<r1>, <r2>, "-|>"), edge(<r2>, <r3>, "-|>"),
    edge(<r3>, <r1>, "-|>", stroke: 0.6pt + cut-colour),
  ),
  diagram(
    spacing: 3.4em, node-stroke: 0.6pt, node-shape: circle, node-inset: 3pt,
    node((0, 0), text(9pt, $r_1$), name: <s1>), node((1, 0), text(9pt, $r_2$), name: <s2>),
    node((0.5, 0.9), text(9pt, $r_3$), name: <s3>, fill: core-colour.lighten(85%)),
    edge(<s1>, <s2>, "-|>"), edge(<s2>, <s3>, "-|>"),
  ),
  small[overwriting pointers], small[union-find],
))[
  Grouping joins the members along their pointers. Overwriting one representative's
  pointer with another's, without first following each to its root, can close a loop
  among representatives (red), and the pointer jumping that follows never ends: a run on
  5 ranks hung this way. A union-find links only roots, so each group stays a tree with
  one root (shaded).
]

#step([Reading past the list], code: local, focus: (13,), window: 2, diagram(
  spacing: (2.2em, 3.2em), node-stroke: 0.6pt, node-inset: 4.5pt,
  node((0, 0), text(9pt, $u_1$), name: <c0>), node((1, 0), text(9pt, $u_2$), name: <c1>),
  node((2, 0), text(9pt, $u_3$), name: <c2>),
  node((3, 0), text(8pt, fill: cut-colour)[$0$], name: <c3>, stroke: (paint: cut-colour, dash: "dashed", thickness: 0.6pt)),
  node((1, -0.55), small[unresolved subtrees], stroke: none),
  node((3, -0.55), text(7.5pt, fill: cut-colour)[one past the end], stroke: none),
  node((1.5, 1), small[subtree of point 0], name: <z>, fill: rgb("#f6d7d2")),
  edge(<c0>, <z>, "-|>", stroke: 0.6pt + cut-colour), edge(<c1>, <z>, "-|>", stroke: 0.6pt + cut-colour),
  edge(<c2>, <z>, "-|>", stroke: 0.6pt + cut-colour),
  edge(<c3>, <z>, "--|>", stroke: 0.6pt + cut-colour, text(7.5pt, fill: cut-colour)[read as the target], label-side: left),
))[
  The shared-memory grouping read one element past its list of unresolved subtrees. The
  stray value was usually 0, so it formed no groups and pointed every unresolved subtree
  at the subtree of point 0, merging them into whatever cluster point 0 was in.
  AddressSanitizer reported the read. Only multithreaded runs reach this line: sequential
  in-place updates collapse cycles on their own.
]

== Picks need not be lightest

The paper's minimum-edge update (`pwrite`) is a compare-and-swap loop. If the compare and
the write are not one atomic operation, two threads proposing to the same subtree can
leave an edge that is not the lightest, or the target of one proposal with the weight of
another. Neither changes the clusters:

#proposition(title: [Borůvka with arbitrary outgoing edges])[
  Suppose each round, every subtree with at least one outgoing edge records _some_
  outgoing edge (not necessarily the lightest), and subtrees are merged along the
  recorded edges with cycles joined as above. Then the procedure terminates, and its
  final subtrees are exactly the connected components of the graph.
] <prop:nonminimal>

#proof[
  _Safety:_ every merge follows an edge of the graph, so no subtree ever spans two
  components. _Progress:_ while some component contains two subtrees, each of them has
  an outgoing edge and records one, and following recorded edges from any of them leads
  into a merge, so the number of subtrees strictly decreases. The procedure stops only
  when no subtree has an outgoing edge, that is, when each subtree is a whole component.
]

Minimality matters for the tree, and for hierarchical clustering, not for the clusters.
The proposition also explains why the clusters do not depend on thread scheduling though
the trees, and the order in which roots are found, do; labels are therefore renumbered at
the end (@sec:numerics).

#takeaway[Borůvka finds the components in $O(log n)$ rounds of "pick an edge, merge
  along the picks", provided the cycles of picks that a directed graph creates are
  detected and joined.]

= Many machines <sec:distributed>

#question[The graph does not fit on one machine. What has to change?]

With $P$ ranks the points are split into contiguous blocks in rank order: with
$n_0 = ceil(n slash P)$, rank $r$ holds points $r n_0, dots, (r + 1) n_0 - 1$, the last
nonempty rank holds the remainder, and trailing ranks may hold nothing (7 points on 6
ranks leave ranks 4 and 5 empty). Each rank stores only its own rows, with global
neighbor ids. The plan: grow trees inside each rank without talking, then treat each tree
as a single vertex and run Borůvka once more across ranks, on a graph far smaller than
the original.

#story()
#step([Blocks and cut edges], code: one-rank, focus: ("in", 1), window: 2, panel("ranks"))[
  The running example on #ex.n_ranks ranks, with points numbered left to right so that
  blocks are strips. Each rank finds its core points alone. An edge to a point of another
  rank is a _cut edge_ #swatch(cut-colour, shape: "line"): there are #ex.n_cut_edges.
]

#step([Local Borůvka, without communication], code: one-rank, focus: (2,), window: 2, panel("local-subtrees"))[
  Each rank runs @alg:local on its block, recording cut edges instead of following them.
  This gives #ex.n_local_subtrees subtrees, some of a single point: a cluster that spans
  ranks is still in pieces.
]

#step([Subtrees become vertices], code: one-rank, focus: (3, 4), window: 2, panel("super-vertices"))[
  Each subtree gets a global id, numbered after those of lower ranks, and the rank that
  numbered it _owns_ it. Each cut edge between points becomes an edge between their
  subtrees, which only the far point's rank can name: #ex.n_links links here.
]

#step([Cut MST: the same clusters], code: one-rank, focus: (5, 6), window: 2, panel("rank-clusters"))[
  Borůvka once more, over the subtrees and across ranks (@alg:cut), and every point takes
  its subtree's root. The clusters are those found on one rank, as the next theorem
  guarantees for any split.
]

#theorem(title: [inexact MST, Theorem 3.6])[
  Partition the vertices into groups. Build a minimum spanning forest of each group
  using only edges inside the group; contract each resulting tree to a super-vertex;
  build a spanning forest of the super-vertices using only the cut edges between groups.
  The union of these forests is generally not a minimum spanning forest of the whole
  graph, but its connected components are those of the graph.
] <thm:inexact>

#proof[
  Any path between two vertices alternates runs of internal edges, whose endpoints the
  group forests already connect, with cut edges, which the forest over super-vertices
  connects. Conversely every forest edge is a graph edge.
]

The theorem is also the property an implementation must keep: *the clusters of core
points do not depend on the number of ranks or on how points are assigned to them.*
Border points are the exception, by DBSCAN's own ambiguity: one with core neighbors in
several clusters prefers a neighbor on its own rank, so its cluster may change with the
rank count.

#scene(width: 40%, diagram(
  spacing: (1.2em, 2.1em),
  node-stroke: 0.6pt,
  node-corner-radius: 3pt,
  node-inset: 5pt,
  node((0, 0), small[Core points], name: <s1>),
  node((0, 1), small[Local MST (OpenMP)], name: <s2>),
  node((0, 2), small[Global subtree ids], name: <s3>),
  node((0, 3), small[Cut edges relabelled], name: <s4>),
  node((0, 4), small[Cut MST], name: <s5>),
  node((0, 5), small[Border points], name: <s6>),
  edge(<s1>, <s2>, "-|>", small[local only], label-side: left),
  edge(<s2>, <s3>, "-|>", small[`Allgather`], label-side: left),
  edge(<s3>, <s4>, "-|>", small[`Alltoallv`], label-side: left),
  edge(<s4>, <s5>, "-|>", small[`Alltoallv`, `Allgatherv` per round], label-side: left),
  edge(<s5>, <s6>, "-|>", small[`Alltoallv` if remote], label-side: left),
))[
  Each rank runs the stages of @alg:driver in order, and only these transitions
  communicate. The communication is collective throughout: `Allgather` and `Allgatherv`
  for counts and roots, `Alltoallv` for edge proposals and pointer queries, and `Gatherv`
  and `Scatterv` for the rare cycle joining on rank 0.

  The paper's analysis gives a local-stage work of $O(n/P dot M + (n/P) log(n/P))$ per
  rank, and cut-stage communication that grows with the number of cut edges per rank and
  with $log P$ rounds. Cut edges increase with the dimension of the data, which is what
  limits strong scaling in high dimensions.
]

== Borůvka across ranks

The cut MST is @alg:local with each step spread over the ranks: the owner of a subtree
collects its proposals, and pointer jumping asks the owner of $p(u)$ for $p(p(u))$. Every
rank updates its pointers at once, which is where the power-of-two cycles of
@fig:cycles appear.

#let cut-mst = (
  require[subtrees, each owned by one rank; cut edges $(u, v, w)$ between subtree ids],
  ensure[the root of every subtree],
  ln(0, [*repeat*]),
  ln(1, [*parallel for* each subtree $u$ this rank touches: its lightest local cut edge], note: [local proposal]),
  ln(1, [send each proposal to the owner of $u$, which keeps the lightest as $E_min [u]$], note: [`Alltoallv`]),
  ln(1, [owners break symmetry between mutual pairs], note: [as in @alg:local]),
  ln(1, [*repeat*], note: [distributed pointer jumping]),
  ln(2, [for each unresolved $u$, ask the owner of $p(u)$ for $p(p(u))$], note: [`Alltoallv` request and reply]),
  ln(2, [keep $u$ unresolved if its chain returns to $u$], note: [power-of-two cycles]),
  ln(1, [*until* every subtree is resolved, *or* the unresolved count stalls]),
  ln(1, [*if* stalled *then* group the unresolved subtrees on rank 0 with a union-find], note: [`Gatherv`, `Scatterv`]),
  ln(1, [share the new roots; relabel cut edges, dropping those inside one subtree], note: [`Allgatherv`]),
  ln(0, [*until* the global number of subtrees does not decrease]),
)
#let cut = (label: <alg:cut>, lines: cut-mst)

#algorithm(caption: [Cut MST: Borůvka over subtrees across ranks (paper's Algorithm 4).],
  pseudocode(cut-mst)) <alg:cut>

#story()
#step([An odd cycle stalls], code: cut, focus: (5, 6, 8, 9), window: 1, grid(
  columns: 3, column-gutter: 1fr, row-gutter: 0.5em,
  pointer-state(triangle, (("a", "b"), ("b", "c"), ("c", "a"))),
  pointer-state(triangle, (("a", "c"), ("b", "a"), ("c", "b"))),
  pointer-state(triangle, (("a", "b"), ("b", "c"), ("c", "a"))),
  small[picks $p$], small[$p^2$], small[$p^4 = p$],
))[
  Three subtrees, owned by any ranks, pick each other in a ring. Every round asks the
  owners for the pointers' pointers, which only rotate: the count of unresolved subtrees
  stops falling, the stall is detected, and rank 0 groups the ring with a union-find.
]

#step([A power-of-two cycle must stay unresolved], code: cut, focus: (5, 6, 7), window: 1, grid(
  columns: 3, column-gutter: 1fr, row-gutter: 0.5em,
  pointer-state(square, (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))),
  pointer-state(square, (("a", "c"), ("b", "d"), ("c", "a"), ("d", "b"))),
  pointer-state(square, (("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"))),
  small[picks $p$], small[$p^2$], small[$p^4$: false "roots"],
))[
  Four subtrees in a ring point at themselves after two rounds and would pass for four
  roots; the cluster would stay in four pieces on some rank counts (one cluster became
  six on 7 ranks). Line 7 keeps a subtree whose chain returned to itself unresolved, so
  the ring stalls like an odd one and is grouped.
]

The Python package runs the same code on a single rank, through a stand-in that
implements each collective for a one-rank communicator, so stages III--V become
bookkeeping. The MPI command-line program runs it on any number of ranks.

#takeaway[Split the points, grow trees locally without talking, then run Borůvka once
  more over the trees: the clusters never depend on the split (@thm:inexact).]

= Details that decide correctness <sec:numerics>

#question[The definitions are exact. Where can code still get them wrong?]

The paper's definitions leave a few decisions to the implementation. Each is fixed by the
code and checked by a test; the first two were wrong in the original implementation.

#invariant(title: [the radius test])[
  A distance equal to $epsilon$ is within $epsilon$ ($<=$, as in @lem:classify and in
  scikit-learn's DBSCAN). Distances and $epsilon$ are compared in single precision, the
  precision of the stored graph: a double-precision $epsilon$ is first rounded to the
  nearest `float32`.
]

#story()
#step([Distances exactly at $epsilon$], code: one-rank, focus: (1,), window: 1, width: 56%, panel("eps-boundary"))[
  On a unit grid with $epsilon = 1$ and $M = #ex.eps_grid.min_samples$, each interior
  point has four neighbors at distance exactly $epsilon$. With $<=$ the interior points
  are core #swatch(core-colour) and the rest border #swatch(border-colour) or noise. The
  strict $<$ of the original implementation finds no core point at all. Grids and
  rounded distances put many points exactly at $epsilon$.
]

#invariant(title: [no finite "no edge yet" weight])[
  The weight that marks a subtree without a candidate edge is $+infinity$. Any finite
  sentinel $s$ silently drops every edge of weight at least $s$, so the result would
  depend on the units of the data.
]

#step([A finite "no edge" weight], code: local, focus: (3,), window: 2, width: 56%, panel("sentinel"))[
  With a finite $s$, edges of weight $s$ or more (faint red) can never be picked, and
  clusters fall apart, here into #ex.sentinel_pieces pieces. The original $s = 10^4$ did
  this to data whose neighbor distances reach $10^4$, such as raw physical units; here
  $s$ is the 60th percentile of the example's edge weights.
]

#invariant(title: [deterministic labels])[
  Clusters are numbered $0, 1, dots$ in order of their lowest-indexed point, and noise is
  $-1$. The core's own numbering follows its roots, which depend on thread scheduling
  (@prop:nonminimal); renumbering makes the labels a function of the partition alone,
  identical for any number of threads.
]

#invariant(title: [invariances])[
  The clustering is unchanged by scaling all distances and $epsilon$ by the same factor,
  by any strictly increasing transformation of distances and $epsilon$ (the algorithm only
  compares them), by permuting the points (which permutes the labels), and by dropping
  graph columns beyond $M - 1$. The clusters of core points are unchanged by the number
  of threads and ranks (@thm:inexact).
]

#invariant(title: [valid input])[
  Each row lists neighbors other than the point, in nondecreasing order of finite,
  nonnegative distance, with $2 <= M <= k + 1$. The algorithm relies on all of it: border
  labelling stops at the first distance beyond $epsilon$ and depends on sorted rows to
  stay inside its row, and an out-of-range index is read as another point's row. Before
  clustering, the library rejects any row with an index out of range, the point itself,
  or a distance that is negative, not finite or out of order, as well as an $M$ outside
  that range.
]

#takeaway[Compare with $<=$ in the graph's precision, never cap edge weights, and
  renumber the labels: then the result depends only on the graph, $epsilon$ and $M$.]

= How we know it is right <sec:verification>

#question[A clustering has no expected value to compare with. How is it tested?]

Against the paper's statements, not against stored outputs. Six kinds of test, each run
on the library, the MPI program and the C API:

#cards(
  ([Reference implementation], [The definitions of @sec:reach computed directly with
    sparse connected components (`tests/reference.py`), written from the paper, never
    from the C++ code. Core points, clusters and noise must match exactly, and each
    border point must join an allowed cluster.]),
  ([Random directed graphs], [Distances with no triangle inequality, rarely mutual
    edges, ties and zeros. Their picks form the cycles of @sec:cycles far more often than
    data does, and they found every defect in the cycle handling.]),
  ([Theorems as tests], [Classes equal scikit-learn's DBSCAN (@lem:classify); clusters
    refine DBSCAN's and coincide when the counts match (@thm:refine); a larger $epsilon$
    only adds core points and merges clusters.]),
  ([Metamorphic tests], [The invariances of @sec:numerics, applied exactly: scaling by
    powers of two (exact in floating point, up to $2^(plus.minus 120)$), integer distances
    under affine and quadratic maps, permutations, truncated graphs.]),
  ([Ranks and threads], [The MPI program on 1 to 7 ranks, with points shuffled (most
    edges cut) or grouped (few cut) and with ranks that own nothing, compared with the
    reference (@thm:inexact); repeated runs and thread counts must agree.]),
  ([Sanitizers], [Everything above under AddressSanitizer and
    UndefinedBehaviorSanitizer. Their builds change memory layout and timing, which
    exposed a data race that optimized builds hid.]),
)

The original implementation, which produced the paper's results, passed its own
experiments. These tests found the following defects in it, all fixed in this
implementation. None is in the method; each is a way code can break the method's
assumptions, and each appears earlier in this companion where it arises.

#figure(
  table(
    columns: (1.25fr, 1.3fr, 1fr),
    align: left,
    table.header[Defect][Effect][Found by],
    [Strict comparison $d < epsilon$], [points exactly at $epsilon$ were not core (grid with $epsilon = 1$: none instead of 16)], [Lemma 3.1 test on a grid],
    [Finite sentinel weight $10^4$], [edges of weight $>= 10^4$ never merged (data scaled by $2^16$: 109 clusters instead of 3)], [scaling by powers of two],
    [Power-of-two cycles in distributed pointer jumping], [clusters split on some rank counts (one cluster became 6 on 7 ranks)], [random graphs over rank counts],
    [Grouping by overwriting representatives], [endless loop in cycle joining (5 ranks)], [random graphs over rank counts],
    [Local cycle grouping read past its array], [subtrees on cycles merged into point 0's cluster], [AddressSanitizer],
    [Loop variable shared by threads], [core points labelled with another thread's root], [sanitized builds, thread sweep],
    [Every rank truncated the output file], [empty labels file on oversubscribed ranks], [MPI tests on CI runners],
  ),
  caption: [Defects found in the original implementation.],
) <tab:defects>

The last two depended only on how threads or ranks were scheduled, which is why they
appeared under sanitizers and on busy CI machines but rarely in ordinary runs:

#story()
#step([A loop variable shared by threads], code: one-rank, focus: (6,), window: 1, table(
  columns: 4,
  align: (center, left, left, center),
  table.header[][thread A][thread B][shared $u$],
  [1], [$u <- R[p]$], [], [$R[p]$],
  [2], [], [$u <- R[q]$], [$R[q]$],
  [3], [#text(fill: cut-colour)[$R[p] <- "root"(u)$]], [], [$R[q]$],
))[
  The loop that gives each core point its cluster used one variable $u$ for all threads.
  Thread B could overwrite it between thread A's read and A's use, and point $p$ received
  the cluster of B's point $q$. Declaring $u$ inside the loop makes it private to each
  iteration.
]

#step([One writer for the output file], table(
  columns: 3,
  align: (center, left, left),
  table.header[][rank 0][rank 2 (late)],
  [1], [open, truncate], [_still computing_],
  [2], [write $n$ labels], [],
  [3], [], [#text(fill: cut-colour)[open, truncate]],
))[
  In the command-line program, outside the algorithm: every rank opened, and so emptied,
  the labels file, but only rank 0 wrote it. A rank that reached the open after rank 0
  had finished left an empty file. Only rank 0 may open it.
]

#note(title: [A lesson for testing parallel graph code])[
  All the cycle defects lived in paths that real data rarely reaches: exact kNN graphs
  of continuous data produce mostly mutual pairs (@sec:cycles), and a few ranks rarely
  produce cycles whose length is a power of two. Randomized inputs that are valid but
  adversarial, swept over thread and rank counts and compared with an independent
  reference, reached them within seconds.
]

#takeaway[Test against the theorems with an independent reference, on adversarial
  random graphs, across thread and rank counts, under sanitizers.]

= Where the time goes <sec:performance>

#question[What does it cost, and what should be computed only once?]

#figure(
  panel("performance", width: 92%),
  caption: [Left: building the kNN graph of 64 million 5-D points with $k = 100$ takes
    about 400 times longer than clustering it. Right: strong scaling of 64 million points
    from 4 to 256 ranks, against the ideal 64. Both from the paper's experiments on TACC's
    Frontera.],
) <fig:performance>

The graph dominates, so compute it once and reuse it across $epsilon$ and $M$
(@sec:reach). Approximate graphs are acceptable: kNN-DBSCAN is defined on whatever graph
it is given, and in the paper's experiments lowering the graph's accuracy to 75 % did not
change the clustering quality. Scaling is limited by cut edges: with more ranks, local
work shrinks but the fraction of cut edges, and the communication per Borůvka round,
grows, and it grows with dimension too. At scale, the paper clusters one billion 3-D
points in under a second on 28K cores, 65 billion 20-D points in under 40 s on 114,688
cores, and runs up to 37× faster than PDBSCAN @patwary2012 on 4 million 20-D points.

On a single node, the clustering stage of the Python package takes about 0.9 s for a
4-million-point graph with 19 neighbors per point on one thread and 0.2--0.26 s on
eight, with no further gain at 16 on a 16-thread laptop. Checking the input graph and
renumbering labels add 5--10 %. Logging costs nothing when disabled.

= Using it <sec:using>

#question[Which $epsilon$ and $M$?]

#scene(width: 52%, panel("kdist"))[
  Choose $M$ first, as for DBSCAN: a larger $M$ smooths the density estimate. Then sort
  the points by the distance to their $(M - 1)$-th neighbor, the number that decides
  whether a point is core (@lem:classify). Dense points form the flat part of the curve
  and sparse points its steep end; an $epsilon$ at the elbow @schubert2017 makes the flat
  part core #swatch(core-colour).
]

Compute a graph with $k >= M - 1$, larger if several $M$ will be tried, and cluster it
for each candidate:

```python
from knndbscan import KNNDBSCAN, knn_dbscan, knn_dbscan_from_graph

labels = knn_dbscan(X, eps=0.05, min_samples=15, n_jobs=-1)            # exact graph
labels = knn_dbscan_from_graph(neighbors, distances, eps=0.05, min_samples=15)
model = KNNDBSCAN(eps=0.05, min_samples=15).fit(X)   # labels_, core_sample_indices_
```

For graphs that do not fit on one node, the MPI program reads a graph file and runs on
any number of ranks:

```sh
mpiexec -n 64 knndbscan -n N -e eps -m M -k K -t threads -i graph.txt -o labels.txt
```

#bibliography("references.yml", title: [References])
