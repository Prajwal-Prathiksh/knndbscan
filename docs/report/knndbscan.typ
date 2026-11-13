#import "template.typ": *
#import "@preview/lovelace:0.3.1": pseudocode-list
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#show: report.with(
  title: [kNN-DBSCAN: algorithm, distributed implementation and correctness],
  subtitle: [A technical companion to Chen, Ruys and Biros, "KNN-DBSCAN: a DBSCAN in high dimensions"],
  authors: ((name: [knndbscan project], affiliation: [github.com/Prajwal-Prathiksh/knndbscan]),),
  date: [November 2026 · knndbscan 0.2],
  abstract: [
    kNN-DBSCAN replaces DBSCAN's range queries by a $k$-nearest-neighbor graph, which can
    be computed approximately and at scale in high dimensions. This report distills the
    method of Chen, Ruys and Biros @paper for readers who want to understand, use or modify
    an implementation without first reading the paper: the definitions and how they relate
    to DBSCAN, the reduction of clustering to a spanning forest, Borůvka's algorithm on a
    directed kNN graph, and the distributed (MPI + OpenMP) algorithm. It adds what the
    paper does not need to say but an implementation must get right: numerical
    conventions, the cycles that directed graphs create in Borůvka's pointer jumping, the
    invariants the code depends on, and how correctness is verified. Every claim about
    the implementation is checked by the project's test suite.
  ],
)

= Introduction

DBSCAN @ester1996 clusters points by density: a point with at least $M$ points within
distance $epsilon$ is a _core point_, core points within $epsilon$ of each other belong to
the same cluster, and points near a cluster's core join it as _border points_; the rest
is _noise_. It needs no number of clusters, finds clusters of any shape and is robust to
outliers. Its cost lies in the $epsilon$-neighborhoods: range queries are cheap with
spatial trees in low dimensions but degrade towards $O(n^2)$ comparisons as the
dimension grows, and every new value of $epsilon$ repeats them.

kNN-DBSCAN @paper keeps DBSCAN's parameters and its classification of points, but
connects core points through a $k$-nearest-neighbor graph ($k$-NNG) instead of
$epsilon$-neighborhoods. A $k$-NNG can be approximated well in high dimensions (randomized
projection trees @xiao2016, as in GOFMM @yu2018), is computed once for all $epsilon$ and
all $M <= k + 1$, and stores $O(n k)$ edges. Clustering the graph then reduces to
computing a spanning forest, which parallelizes with Borůvka's algorithm in shared and
distributed memory.

The report follows the path from definitions to code:
@sec:dbscan and @sec:knn define both methods and relate them;
@sec:forest reduces clustering to a spanning forest and develops Borůvka's algorithm on a
directed graph; @sec:distributed distributes it over MPI ranks;
@sec:numerics collects the numerical conventions and invariants an implementation must
keep; @sec:verification describes how correctness is verified, with the defects this
found in the original implementation; @sec:performance summarizes performance.

#note[
  Numbers such as "Theorem 3.4" refer to the paper. Statements here are restated in a
  single notation and sometimes split or sharpened; the proofs are sketches that convey
  why a result holds. The code is organized by the same stages: `docs/ALGORITHM.md` maps
  each one to its file and function.
]

== Notation

#figure(
  table(
    columns: (auto, 1fr),
    align: (center, left),
    table.header[Symbol][Meaning],
    $cal(I)$, [the set of $n$ points],
    $d(p, q)$, [distance between points $p$ and $q$ (Euclidean in the Python package)],
    $epsilon, M$, [radius and minimum number of points (`eps`, `min_samples`); $M$ counts the point itself],
    $cal(N)_epsilon (p)$, [$epsilon$-neighborhood ${q in cal(I) : d(p, q) <= epsilon}$, including $p$],
    $cal(N)_M (p)$, [the $M$ nearest points to $p$, including $p$],
    $k$, [neighbors stored per point in the $k$-NNG, excluding the point itself],
    $cal(I)_"core", cal(I)_"border", cal(I)_"noise"$, [the three classes of points],
    $P$, [number of MPI ranks],
  ),
  caption: [Notation.],
)

= DBSCAN <sec:dbscan>

#definition(title: [classification of points, Def. 3.1])[
  Given $epsilon$ and $M$, a point $p$ is a _core point_ if $|cal(N)_epsilon (p)| >= M$. A
  point that is not core is a _border point_ if some core point $q$ has
  $p in cal(N)_epsilon (q)$, and _noise_ otherwise.
] <def:classes>

#definition(title: [reachability and DBSCAN clusters, Defs. 3.2--3.5])[
  $q$ is _directly $epsilon$-reachable_ from $p$ if $p$ is core and $q in cal(N)_epsilon (p)$;
  _$epsilon$-reachable_ if a chain of core points $p = p_0, p_1, dots, p_m$ links them by
  direct reachability, ending in $q$. A _DBSCAN cluster_ is a nonempty set $C$ that is
  closed under $epsilon$-reachability (if $p in C$ is core and $q$ is reachable from $p$,
  then $q in C$) and connected (any two members are reachable from a common core point).
]

On core points, $epsilon$-reachability is an equivalence relation, so the core points of
the clusters are its equivalence classes: the connected components of the graph that
joins core points at distance at most $epsilon$. Border points are not uniquely placed: a
border point within $epsilon$ of core points of two clusters belongs to either. This
ambiguity is inherent to DBSCAN and carries over to kNN-DBSCAN.

= kNN-DBSCAN <sec:knn>

== Classifying points from the $k$-NNG

The first observation is that the classification of @def:classes needs only the $M$
nearest neighbors of each point.

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

Consequently a $k$-NNG with $k >= M - 1$ classifies every point exactly as DBSCAN does,
and the classification is the same whatever larger $k$ was stored. In the stored rows,
which list each point's neighbors in increasing distance without the point itself, $p$
is core when the distance in column $M - 1$ is at most $epsilon$.

== Reachability through the graph

DBSCAN links core points that are within $epsilon$ of each other. kNN-DBSCAN links them
only through the graph:

#definition(title: [$M$-reachability and kNN-DBSCAN clusters, Defs. 3.6--3.9])[
  $q$ is _directly $M$-reachable_ from $p$ if $p$ is core and $q in cal(N)_M (p)$ or
  $p in cal(N)_M (q)$. $M$-reachability, $M$-connectedness and _kNN-DBSCAN clusters_ are
  defined from it exactly as their $epsilon$ counterparts.
] <def:mreach>

The relation is symmetrized: an edge counts if either endpoint lists the other. Two
facts make it well behaved.

#lemma(title: [Lemmas 3.2 and 3.3])[
  On core points, $M$-reachability is an equivalence relation, and it implies
  $epsilon$-reachability.
] <lem:equivalence>

#proof[
  Symmetry comes from the symmetrized definition; reflexivity and transitivity from
  concatenating chains. If $p$ is core then every $q in cal(N)_M (p)$ satisfies
  $d(p, q) <= epsilon$ by @lem:classify, so each $M$-step is an $epsilon$-step.
]

So the core points of kNN-DBSCAN clusters are the connected components of the graph
$G_(M, "core")$ whose vertices are the core points and whose edges join $p$ to its
first $M - 1$ neighbors when both are core. No distance test is needed on these edges:
they are within $epsilon$ automatically.

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

kNN-DBSCAN can therefore _split_ a cluster that DBSCAN joins through a sparse bridge,
where two core points are within $epsilon$ but not among each other's $M - 1$ nearest
neighbors, and never _merges_ clusters that DBSCAN separates. In the paper's
experiments, and in this project's tests on blobs, moons, anisotropic and 20-dimensional
data, the clusterings agree on the core points whenever the counts agree, as part (2)
guarantees.

#note(title: [Consequences for the graph])[
  Only the first $M - 1$ columns of each row are ever read: they decide core points
  (column $M - 1$), cluster edges (columns $1, dots, M - 1$) and border candidates
  (@lem:classify, part 2). A graph stored with $k$ neighbors therefore serves every
  $M <= k + 1$ and every $epsilon$ without recomputation, which is what makes
  parameter search cheap: the graph is the expensive part (@sec:performance).
]

= Clusters as spanning forests <sec:forest>

== From components to spanning trees

#theorem(title: [Theorem 3.5])[
  The connected components of a minimum spanning forest of $G_(M, "core")$ are the
  kNN-DBSCAN clusters of the core points (and those of $G_(epsilon, "core")$ are the
  DBSCAN clusters).
]

A spanning forest has the same components as the graph, so the statement is immediate;
its value is algorithmic. Spanning forests can be built in parallel with Borůvka's
algorithm, and a _minimum_ spanning forest additionally supports hierarchical
clustering. For clustering alone minimality is not needed, a fact the implementation
relies on (@prop:nonminimal).

== Borůvka's algorithm

Borůvka's algorithm grows all trees at once. Each _subtree_ (initially a single core
point) selects its lightest edge to another subtree; the selected edges merge subtrees;
the round repeats until no subtree has an edge left. Each round at least halves the
number of subtrees that can still merge, so $O(log n)$ rounds suffice. A round has four
steps, shown in @alg:local for one shared-memory process:

/ Minimum edges: every core point proposes its lightest edge to another subtree, and each
  subtree keeps the lightest proposal (`pwrite`, Algorithm 3 of the paper).
/ Breaking symmetry: each subtree $u$ points to the target $p(u)$ of its edge, or to itself
  if it has none. Two subtrees that chose each other point at the one with the smaller id.
/ Pointer jumping: $p(u) <- p(p(u))$ until nothing changes; then every subtree points to
  the root of its new tree.
/ Relabelling: each core point takes the root of its subtree.

#algorithm(caption: [Local Borůvka over the core points of one process (paper's Algorithm 2).])[
  #pseudocode-list[
    + *input:* core points $C$; sorted neighbor rows; $R[i] <- i$ for $i in C$
    + $L[i] <- 1$ for $i in C$ #h(1fr) _next neighbor column of $i$ to inspect_
    + *repeat*
      + $E_min [u] <- ("none", +infinity)$ for every subtree $u$
      + *for each* $i in C$ *in parallel*
        + *while* $L[i] <= M - 1$
          + $q <-$ neighbor $L[i]$ of $i$
          + *if* $q$ is stored on another rank *then* record cut edge; $L[i] <- L[i] + 1$
          + *else if* $q in C$ *and* $R[q] != R[i]$ *then* `pwrite`$(E_min [R[i]], R[q], d(i, q))$; *break*
          + *else* $L[i] <- L[i] + 1$ #h(1fr) _non-core, or already in the same subtree_
      + $p(u) <- E_min [u]."target"$, or $u$ if none; *if* $p(p(u)) = u$ *and* $u < p(u)$ *then* $p(u) <- u$
      + *repeat* $p(u) <- p(p(u))$ *until* no pointer changes
      + join the subtrees left on cycles (@sec:cycles)
      + $R[i] <- p(R[i])$ for every $i in C$
    + *until* the number of subtrees does not decrease
  ]
] <alg:local>

Two details matter for cost. Rows are sorted, so the first eligible neighbor of a point
is its lightest outgoing edge, and the scan stops there. And an edge that has become
internal to a subtree stays internal, so the column pointer $L[i]$ only moves forward:
over all rounds each of the $n(M-1)$ edges is inspected a bounded number of times.

== Cycles in a directed graph <sec:cycles>

In an undirected graph with distinct weights, the selected edges contain no cycles except
mutual pairs, which symmetry breaking resolves: following selections strictly decreases
the weight. A kNN graph is _directed_: $q in cal(N)_M (p)$ does not imply
$p in cal(N)_M (q)$, so subtree $u$ sees only the edges its own points list. The weights
along a chain of selections need not decrease, and longer cycles appear (@fig:cycle).
Ties in the weights, frequent with duplicated or quantized data, create them too.

#figure(
  diagram(
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
  ),
  caption: [The directed graph of the paper's Fig. 4 ($k = 2$). Each vertex's lightest
    out-edge is drawn: $V_2$ does not list $V_1$, so instead of the mutual pair
    $V_1 harpoons.rtlb V_2$ an undirected graph would give, the selections form a 4-cycle,
    and pointer jumping alone cannot terminate correctly.],
) <fig:cycle>

Pointer jumping replaces $p$ by $p compose p$ in every round. A vertex on a tree reaches
its root after $ceil(log_2 "depth")$ rounds and stays there. A vertex on a cycle of
length $ell$ never reaches a root, but what it does next depends on $ell$:

#figure(
  table(
    columns: (auto, 1fr, 1fr, auto, auto),
    align: (left, center, center, center, left),
    table.header[Cycle][1 round: $p^2$][2 rounds: $p^4$][3 rounds: $p^8$][Result],
    [$a -> b -> c -> a$], [$a -> c, b -> a, c -> b$], [$a -> b, b -> c, c -> a$], [$= p^2$], [never resolved],
    [$a -> b -> c -> d -> a$], [$a -> c, b -> d, dots$], [*$a -> a, b -> b, dots$*], [$= p^4$], [four "roots"],
  ),
  caption: [Pointer jumping on cycles. After $t$ rounds each pointer is $p^(2^t)$, which is
    the identity once $ell$ divides $2^t$: on a cycle whose length is a power of two,
    every member points at itself, which is exactly how a root looks.],
) <tab:jumping>

#pitfall(title: [Power-of-two cycles])[
  An implementation that detects cycles only by the number of unresolved subtrees no
  longer decreasing catches odd cycles, but on a cycle of length 2, 4, 8, ... every
  member "resolves" to itself and becomes a root of its own. The cycle is never merged,
  the next round selects the same edges, no subtree merges, and Borůvka stops early with
  a cluster split into pieces. A subtree that was not a root and whose chain returns to
  itself is on a cycle and must stay unresolved, so that cycle breaking sees it.
  Sequential in-place updates hide the problem, because the first member to point at
  itself becomes the root of the others; synchronous updates (the distributed
  algorithm) expose it.
]

The subtrees left unresolved form cycles with trees hanging off them. They are joined
by grouping them along their pointers into weakly connected groups and making one member
of each group the root of all of it.

#pitfall(title: [Grouping must use a union-find])[
  Grouping by overwriting one representative's pointer with another's, without first
  following each to its root, can link representatives into a new cycle of length three
  or more, on which the pointer jumping that follows loops forever. A union-find with
  path compression cannot.
]

== Minimum edges are not needed

The paper's minimum-edge update (`pwrite`) is a compare-and-swap loop. If the compare
and the write are not one atomic operation, two threads updating the same subtree can
leave an edge that is not the lightest, or even a target from one write with the
weight of another. Neither affects the clusters:

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

Minimality matters for the tree (and for hierarchical clustering), not for the
clusters. The proposition also explains why the result does not depend on thread
scheduling, though the spanning tree, and with it the order in which roots are found,
does. Labels are therefore renumbered at the end (@sec:numerics).

= The distributed algorithm <sec:distributed>

== Data layout

With $P$ ranks, the points are split into contiguous blocks in rank order: with
$n_0 = ceil(n slash P)$, rank $r$ holds points $r n_0, dots, (r+1) n_0 - 1$, the last
nonempty rank holds the remainder, and trailing ranks may hold nothing (7 points on 6
ranks leave ranks 4 and 5 empty). Each rank stores only its own rows, with global
neighbor ids. An edge to a point of another rank is a _cut edge_.

== The inexact minimum spanning tree

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

For kNN-DBSCAN the groups are the ranks. The local stage needs no communication, and
the cut stage works on far fewer vertices (subtrees) than points. The theorem also states
the property an implementation must satisfy: *the clusters of core points do not depend
on the number of ranks or on how points are assigned to them.*

#figure(
  diagram(
    spacing: (1.2em, 2.3em),
    node-stroke: 0.6pt,
    node-corner-radius: 3pt,
    node-inset: 6pt,
    node((0, 0), [I. Core points], name: <s1>),
    node((0, 1), [II. Local MST (OpenMP)], name: <s2>),
    node((0, 2), [III. Global subtree ids], name: <s3>),
    node((0, 3), [IV. Cut edges relabelled], name: <s4>),
    node((0, 4), [V. Cut MST (Borůvka over subtrees)], name: <s5>),
    node((0, 5), [VI. Border points], name: <s6>),
    edge(<s1>, <s2>, "-|>", [local only], label-side: left, label-sep: 0.8em),
    edge(<s2>, <s3>, "-|>", [`Allgather` subtree counts], label-side: left, label-sep: 0.8em),
    edge(<s3>, <s4>, "-|>", [`Alltoallv` ask owners of far endpoints], label-side: left, label-sep: 0.8em),
    edge(<s4>, <s5>, "-|>", [`Alltoallv`, `Allgatherv` each round], label-side: left, label-sep: 0.8em),
    edge(<s5>, <s6>, "-|>", [`Alltoallv` for remote neighbors], label-side: left, label-sep: 0.8em),
  ),
  caption: [Stages of `knndbscan()` on each rank, with the communication each needs.],
) <fig:stages>

== Stages

/ I. Core points: local: column $M - 1$ against $epsilon$.
/ II. Local MST: @alg:local over the rank's own core points and the edges between them,
  with OpenMP threads. Edges to other ranks are collected as cut edges.
/ III. Subtree ids: the number of subtrees of each rank is gathered (`Allgather`), and each
  rank numbers its subtrees in a contiguous block of global ids. The rank that numbered
  a subtree _owns_ it for the rest of the algorithm.
/ IV. Cut edges: each cut edge $(i, j)$ is rewritten as an edge between the subtrees of
  $i$ and $j$. The subtree of $j$ is known only to $j$'s rank, which answers through an
  `Alltoallv` exchange of requests and replies.
/ V. Cut MST: Borůvka over the subtrees (@alg:cut).
/ VI. Border points: each non-core point takes the cluster of its first core neighbor
  within $epsilon$, preferring one on its own rank and otherwise asking the owners of its
  remote neighbors. Either choice is valid (@sec:dbscan); it is why border assignments,
  unlike clusters of core points, may differ between rank counts.

#algorithm(caption: [Cut MST: Borůvka over subtrees across ranks (paper's Algorithm 4).])[
  #pseudocode-list[
    + *input:* the subtrees each rank touches; cut edges $(u, v, w)$ between subtree ids
    + *repeat*
      + every rank: for each subtree $u$ it touches, the lightest local cut edge from $u$
      + `Alltoallv`: send the proposals to the owner of $u$, which keeps the lightest as $E_min [u]$
      + owners break symmetry between mutual pairs, as in @alg:local
      + *repeat* #h(1fr) _distributed pointer jumping_
        + for unresolved $u$: ask the owner of $p(u)$ for $p(p(u))$ (`Alltoallv` request and reply)
        + a subtree whose chain returns to itself stays unresolved (@tab:jumping)
      + *until* every subtree is resolved, *or* the unresolved count stalls
      + *if* stalled *then* `Gatherv` the unresolved subtrees and pointers to rank 0, group them with a union-find, `Scatterv` the roots
      + `Allgatherv` the new root of every subtree; relabel cut edges and drop those inside one subtree
    + *until* the global number of subtrees does not decrease
  ]
] <alg:cut>

#implementation(title: [Communication and cost])[
  The communication is collective throughout: `Allgather` and `Allgatherv` for counts and
  roots, `Alltoallv` for edge proposals and pointer queries, `Gatherv` and `Scatterv` for
  the rare cycle breaking on rank 0. The paper's analysis gives a local-stage work of
  $O(n/P dot M + (n/P) log(n/P))$ per rank, and cut-stage communication that grows with
  the number of cut edges per rank and with $log P$ rounds; cut edges increase with the
  dimension of the data, which is what limits strong scaling in high dimensions.
]

The library used from Python runs the same code on a single rank, through a stand-in
that implements each collective for a one-rank communicator, so stages III--V degenerate
to bookkeeping. The MPI command-line program runs it on any number of ranks.

= Numerical conventions and invariants <sec:numerics>

The paper's definitions leave a few decisions to the implementation. Each of the
following is fixed by the code and checked by a test.

#invariant(title: [the radius test])[
  A distance equal to $epsilon$ is within $epsilon$ ($<=$, as in @lem:classify and in
  scikit-learn's DBSCAN). Distances and $epsilon$ are compared in single precision, the
  precision of the stored graph: a double-precision $epsilon$ is first rounded to the
  nearest `float32`.
]

#invariant(title: [no finite "no edge yet" weight])[
  The weight that marks a subtree without a candidate edge is $+infinity$. Any finite
  sentinel $s$ silently drops every edge of weight at least $s$, so the result would
  depend on the units of the data.
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

= Verifying an implementation <sec:verification>

Clustering output has no simple "expected value", so the project checks the
implementation against the paper's statements rather than against fixed outputs.

/ Reference implementation: the definitions of @sec:knn computed directly with sparse
  connected components (`tests/reference.py`), written from the paper, never from the
  C++ code. The library must reproduce its core points, clusters and noise exactly, and
  give each border point an allowed cluster.
/ Random directed graphs: graphs whose distances obey no triangle inequality and whose
  edges are rarely mutual, with ties and zero distances. They make Borůvka meet the
  cycles of @sec:cycles far more often than real data does, and they were the most
  effective tests: they found every defect in the cycle handling.
/ Theorems as tests: core, border and noise points equal scikit-learn's DBSCAN
  (@lem:classify), clusters refine DBSCAN's and coincide when the counts match
  (@thm:refine), and growing $epsilon$ only adds core points and merges clusters.
/ Metamorphic tests: the invariances of @sec:numerics, applied exactly: scaling by powers
  of two (exact in floating point, up to $2^(plus.minus 120)$), integer distances under
  affine and quadratic maps, random permutations, truncated graphs.
/ Ranks and threads: the MPI program on 1 to 7 ranks, with points shuffled (most edges
  cut) or grouped by cluster (few cut), with ranks that own no points, compared with the
  reference (@thm:inexact); repeated runs and thread counts must agree.
/ Sanitizers: the library, the MPI program and C tests built with AddressSanitizer and
  UndefinedBehaviorSanitizer and run against all of the above. Sanitized builds change
  memory layout and timing, which exposed a data race that optimized builds hid.

The original implementation, which produced the paper's results, passed its own
experiments; the tests above found the following defects, all now fixed. None is in the
method; each is a way the method's assumptions can be broken in code.

#figure(
  table(
    columns: (1.25fr, 1.3fr, 1fr),
    align: left,
    table.header[Defect][Effect][Found by],
    [Strict comparison $d < epsilon$], [points exactly at $epsilon$ were not core (grid with $epsilon = 1$: none instead of 16)], [Lemma 3.1 test on a grid],
    [Finite sentinel weight $10^4$], [edges of weight $>= 10^4$ never merged (data scaled by $2^16$: 109 clusters instead of 3)], [scaling by powers of two],
    [Power-of-two cycles in distributed pointer jumping], [clusters split on some rank counts (one cluster became 6 on 7 ranks)], [random graphs over rank counts],
    [Grouping by overwriting representatives], [endless loop in cycle breaking (5 ranks)], [random graphs over rank counts],
    [Local cycle grouping read past its array], [subtrees on cycles merged into point 0's cluster], [AddressSanitizer],
    [Loop variable shared by threads], [core points labelled with another thread's root], [sanitized builds, thread sweep],
    [Every rank truncated the output file], [empty labels file on oversubscribed ranks], [MPI tests on CI runners],
  ),
  caption: [Defects found in the original implementation by the tests of @sec:verification.],
) <tab:defects>

#note(title: [A lesson for testing parallel graph code])[
  All the cycle defects lived in paths that real data rarely reaches: exact kNN graphs
  of continuous data produce mostly mutual pairs, and a few ranks rarely produce cycles
  whose length is a power of two. Randomized inputs that are valid but adversarial, swept
  over thread and rank counts and compared with an independent reference, reached them
  within seconds.
]

= Performance <sec:performance>

The paper reports, on TACC's Frontera, one billion 3-D points clustered in under a
second on 28K cores, 65 billion 20-D points in under 40 s on 114,688 cores, and runs up to
37× faster than PDBSCAN @patwary2012 on 4 million 20-D points. Strong scaling of 64
million points from 4 to 256 ranks gives speedups of 13.8 (3-D), 16.8 (5-D) and 22.2
(32-D). Two observations shape how the method is used:

- *The graph dominates.* Building the $k$-NNG of 64 million 5-D points with $k = 100$ takes
  about 400 times longer than clustering it, so the graph should be computed once and
  reused across $epsilon$ and $M$ (@sec:knn). Approximate graphs are acceptable: kNN-DBSCAN is
  defined on whatever graph it is given, and in the paper's experiments lowering the
  graph's accuracy to 75 % did not change the clustering quality.
- *Cut edges limit scaling.* With more ranks, local work shrinks but the fraction of cut
  edges, and the communication per Borůvka round, grows; it grows with dimension too.

On a single node, the clustering stage of the Python package takes about 0.9 s for a
4-million-point graph with 19 neighbors per point on one thread and 0.2--0.26 s on
eight, with no further gain at 16 on a 16-thread laptop. Checking the input graph and
renumbering labels add 5--10 %. Logging costs nothing when disabled.

= Using it

Choose $M$ first, as for DBSCAN (larger $M$ smooths the density estimate); then choose
$epsilon$ from the sorted distances to each point's $(M-1)$-th neighbor, at the elbow of
that curve @schubert2017. Compute a graph with $k >= M - 1$, larger if several $M$ will
be tried, and cluster it for each candidate:

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
