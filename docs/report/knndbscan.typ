#import "template.typ": *
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
@app:pictures retells the algorithm as a storyboard on a small example: each step shows
the pseudocode with the lines in question highlighted, beside a picture of what they do,
for every stage and for every defect.

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
] <thm:mst>

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

#let local-boruvka = (
  require[core points $C$ of this process; neighbor rows sorted by distance],
  ensure[$R[i]$, the root of the subtree that contains each $i in C$],
  ln(0, [$R[i] <- i$ and $L[i] <- 1$ for every $i in C$], note: [$L[i]$: next column of row $i$]),
  ln(0, [*repeat*]),
  ln(1, [$E_min [u] <- ("none", +infinity)$ for every subtree $u$]),
  ln(1, [*parallel for* $i in C$], note: [minimum edges]),
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

#algorithm(caption: [Local Borůvka over the core points of one process (paper's Algorithm 2).], pseudocode(local-boruvka)) <alg:local>

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

@alg:driver is one rank's whole run. Stages I and II need no communication; stage III
gives every subtree a global id, numbering each rank's subtrees after those of lower
ranks, and the rank that numbered a subtree _owns_ it from then on; stage IV rewrites
each cut edge between points as an edge between their subtrees, which only the far
point's rank knows. Border points may join any adjacent cluster (@sec:dbscan); one on
several ranks prefers a core neighbor on its own rank, which is why border assignments,
unlike clusters of core points, may differ between rank counts.

#let driver = (
  require[this rank's rows of the kNN graph, sorted, without the point itself; $epsilon$; $M$],
  ensure[a cluster label for each of this rank's points, $-1$ for noise],
  ln(0, [$C <- {i : "distance in column" M - 1 "of row" i <= epsilon}$], note: [I. core points (@lem:classify)]),
  ln(0, [$R$, cut edges $<-$ local Borůvka over $C$], note: [II. @alg:local]),
  ln(0, [number this rank's subtrees after those of lower ranks], note: [III. `Allgather`]),
  ln(0, [rewrite each cut edge $(i, j)$ as $(R[i], R[j])$], note: [IV. `Alltoallv` to $j$'s rank]),
  ln(0, [roots $<-$ cut MST over the subtrees], note: [V. @alg:cut]),
  ln(0, [$R[i] <-$ root of $R[i]$ for every $i in C$]),
  ln(0, [*parallel for* each non-core point $i$], note: [VI. border points]),
  ln(1, [$q <-$ first core neighbor of $i$ within $epsilon$, preferring one on this rank], note: [`Alltoallv` if remote]),
  ln(1, [$R[i] <- R[q]$ if $q$ exists, otherwise $-1$], note: [noise]),
)

#algorithm(caption: [kNN-DBSCAN on one rank: `knndbscan()` (paper's Algorithm 1).], pseudocode(driver)) <alg:driver>

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

#algorithm(caption: [Cut MST: Borůvka over subtrees across ranks (paper's Algorithm 4).], pseudocode(cut-mst)) <alg:cut>

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
experiments; the tests above found the following defects, all fixed in this implementation. None is in the
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


// ------------------------------------------------------------------------------- appendix
#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1", supplement: [Appendix])

// Numbers about the example, written by figures/make_figures.py with the panels.
#let ex = json("figures/example.json")
#let panel(name) = image("figures/" + name + ".svg", width: 100%)
#let local = (label: <alg:local>, lines: local-boruvka)
#let one-rank = (label: <alg:driver>, lines: driver)
#let cut = (label: <alg:cut>, lines: cut-mst)
#let red = rgb("#c0392b")

// Pointer states drawn as a small graph: `pointers` maps each node to its target; a node
// pointing at itself looks like a root, and is drawn shaded with a loop.
#let pointer-state(positions, pointers) = {
  let roots = pointers.filter(((u, v)) => u == v).map(((u, v)) => u)
  diagram(
    spacing: (2.3em, 2.3em),
    node-stroke: 0.6pt,
    node-shape: circle,
    node-inset: 3pt,
    ..positions.pairs().map(((name, pos)) => node(
      pos, text(9pt)[$#name$], name: label(name), fill: if name in roots { rgb("#f6d7d2") } else { white },
    )),
    ..pointers.map(((u, v)) => if u == v {
      edge(label(u), label(u), "-|>", bend: 130deg, stroke: 0.6pt + red)
    } else {
      edge(label(u), label(v), "-|>", bend: 14deg, stroke: 0.6pt)
    }),
  )
}
#let square = (a: (0, 0), b: (1, 0), c: (1, 1), d: (0, 1))
#let triangle = (a: (0, 0), b: (1, 0), c: (0.5, 0.9))
#let states(..pairs) = grid(
  columns: pairs.pos().len(),
  column-gutter: 1fr,
  row-gutter: 0.5em,
  ..pairs.pos().map(((d, _)) => align(center + horizon, d)),
  ..pairs.pos().map(((_, c)) => align(center, text(8pt, c))),
)

= The algorithm in pictures <app:pictures>

This appendix runs the algorithm on a small example and tells it as a sequence of steps.
Each step shows the pseudocode on the left with the lines the step is about highlighted,
and on the right what those lines do to the example, so the code and its effect can be
read together. The example has #ex.n_points points in three blobs with $M = #ex.min_samples$
and $epsilon = #ex.eps$. Its panels are computed, not drawn: `figures/make_figures.py`
replays Borůvka's rounds exactly as @alg:local states them, stops unless the replay finds
the library's clusters, and writes the numbers quoted here.

#step-counter.update(0)
== One rank, from graph to labels

#step([The input: a directed kNN graph], code: one-rank, focus: ("in",), panel("knn-graph"))[
  Each point lists its $M - 1 = #(ex.min_samples - 1)$ nearest neighbors in order of
  distance, itself excluded: one arrow per entry. The graph is _directed_: $q$ among $p$'s
  neighbors does not put $p$ among $q$'s, which is why an isolated point can still have
  arrows leaving it.
]

#step([Core points], code: one-rank, focus: (1,), panel("core-points"))[
  A point is core when its $(M - 1)$-th neighbor is within $epsilon$: its $epsilon$-disk
  (dashed) holds $M$ points, itself included. The test reads one column of each row and
  needs no communication. Here #ex.n_core of the #ex.n_points points are core; the grey
  disk holds too few points.
]

#step([Clusters of core points], code: one-rank, focus: (2, 3, 4, 5, 6), panel("components"))[
  Keep the edges between core points and forget their direction: the components of this
  graph, $G_(M, "core")$, are the clusters (@thm:mst). Lines 2--6 find them as a spanning
  forest, on one rank by Borůvka (@app:boruvka) and on several by local trees joined across
  ranks (@app:ranks). There are #ex.n_clusters.
]

#step([Border points and noise], code: one-rank, focus: (7, 8, 9), panel("labels"))[
  Each remaining point takes the cluster of its first core neighbor within $epsilon$
  (ringed; one is joined to the neighbor it follows), or is noise, $-1$, if it has none.
  Here #ex.n_border points are border points and #ex.n_noise are noise.
]

#step([Not DBSCAN], panel("dbscan"))[
  DBSCAN on the same data finds #ex.n_dbscan_clusters clusters. The red pairs are core
  points within $epsilon$ of each other that are not among each other's $M - 1$ nearest
  neighbors: DBSCAN links them, kNN-DBSCAN does not, so it keeps two blobs apart. The
  core points and border points are the same in both methods; only the links differ,
  and kNN-DBSCAN's clusters always refine DBSCAN's (@thm:refine).
]

#step-counter.update(0)
== Borůvka, round by round <app:boruvka>

#step([Round 1: every subtree picks its lightest edge], code: local, focus: (1, 3, 4, 5, 6, 7, 8, 9), panel("boruvka-1"))[
  Every core point starts as a subtree of its own (dark). It scans its row from the left
  and stops at the first core neighbor in another subtree; rows are sorted, so that is
  its lightest edge (arrows). Red arrows mark pairs of subtrees that chose each other:
  there are #ex.mutual.at(0).
]

#step([Round 1: merge along the chosen edges], code: local, focus: (10, 11, 12, 13, 14), panel("boruvka-merged-1"))[
  Each subtree points to the subtree its edge reaches. In a mutual pair the smaller id
  points at itself and becomes a root; pointer jumping then takes every subtree to the
  root of its tree, and relabelling hands that root to every point. The
  #ex.rounds.at(0) subtrees become #ex.rounds.at(1), and each new subtree holds exactly
  one mutual pair, its root. When selections close a longer cycle instead, nothing is a
  root: @app:defects shows what then happens.
]

#step([Round 2], code: local, focus: (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), panel("boruvka-2"))[
  The same steps on the merged subtrees (coloured; their edges are the tree so far). A
  point whose next neighbor is now in its own subtree moves past it, and $L[i]$ never moves
  back, so every row is scanned once over all rounds. The #ex.rounds.at(1) subtrees
  become #ex.rounds.at(2).
]

#step([Round 3], code: local, focus: (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), panel("boruvka-3"))[
  #ex.rounds.at(2) subtrees become #ex.rounds.at(3). Each round at least halves the
  subtrees that can still merge, so the rounds grow like $log n$.
]

#step([No edge left: the clusters], code: local, focus: (15,), panel("boruvka-end"))[
  The next round finds no edge between subtrees, the number of subtrees stays at
  #ex.rounds.at(3), and the loop ends. The coloured edges are a spanning forest of
  $G_(M, "core")$ with one tree per cluster.
]

#step-counter.update(0)
== Across ranks <app:ranks>

#step([Blocks and cut edges], code: one-rank, focus: ("in", 1), panel("ranks"))[
  The points are split into #ex.n_ranks contiguous blocks of rows, one per rank (here
  numbered left to right, so blocks are strips). Each rank finds its core points alone.
  An edge to a point of another rank is a _cut edge_ (red): there are #ex.n_cut_edges.
]

#step([Local Borůvka, without communication], code: one-rank, focus: (2,), panel("local-subtrees"))[
  Each rank runs @alg:local on its block and records cut edges instead of following them.
  The result is #ex.n_local_subtrees subtrees, some a single point: a cluster that spans
  ranks is still in pieces.
]

#step([Subtrees become vertices], code: one-rank, focus: (3, 4), panel("super-vertices"))[
  Each subtree gets a global id, owned by the rank that numbered it, and each cut edge
  between points becomes an edge between their subtrees (#ex.n_links links here). The
  cut MST works on this much smaller graph.
]

#step([Cut MST: the same clusters], code: one-rank, focus: (5, 6), panel("rank-clusters"))[
  Borůvka once more, over the subtrees and across ranks (@alg:cut), and every point takes
  its subtree's root. The clusters are those of one rank: by @thm:inexact the split never
  changes them, however the points are distributed.
]

#step-counter.update(0)
== Where implementations go wrong <app:defects>

Each step below is a row of @tab:defects, shown at the line where it arises.

#step([A cycle has no root], code: cut, focus: (5, 6, 8, 9), states(
  (pointer-state(triangle, (("a", "b"), ("b", "c"), ("c", "a"))), [selections $p$]),
  (pointer-state(triangle, (("a", "c"), ("b", "a"), ("c", "b"))), [$p^2$]),
  (pointer-state(triangle, (("a", "b"), ("b", "c"), ("c", "a"))), [$p^4 = p$]),
))[
  In a directed graph three subtrees can each choose the next: $a -> b -> c -> a$. No pair
  is mutual, so nothing becomes a root, and pointer jumping rotates the pointers forever.
  The number of unresolved subtrees stops falling; the stall is detected and the cycle is
  grouped on rank 0.
]

#step([A power-of-two cycle looks like roots], code: cut, focus: (5, 6, 7), states(
  (pointer-state(square, (("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))), [selections $p$]),
  (pointer-state(square, (("a", "c"), ("b", "d"), ("c", "a"), ("d", "b"))), [$p^2$]),
  (pointer-state(square, (("a", "a"), ("b", "b"), ("c", "c"), ("d", "d"))), [$p^4$: four "roots"]),
))[
  On a 4-cycle, two rounds of jumping take every pointer to its own node, and a node
  pointing at itself is what a root looks like (shaded). Taken as roots, the four
  subtrees never merge, the next round selects the same edges, and the cluster stays in
  four pieces. Line 7 is the fix: a subtree that was not a root and whose chain returns to
  itself is on a cycle and stays unresolved, so the cycle is grouped.
]

#step([Grouping must link roots], code: cut, focus: (9,), grid(
  columns: 2,
  column-gutter: 2.5em,
  row-gutter: 0.4em,
  diagram(
    spacing: 3.4em, node-stroke: 0.6pt, node-shape: circle, node-inset: 3pt,
    node((0, 0), text(9pt, $r_1$), name: <r1>), node((1, 0), text(9pt, $r_2$), name: <r2>),
    node((0.5, 0.9), text(9pt, $r_3$), name: <r3>),
    edge(<r1>, <r2>, "-|>"), edge(<r2>, <r3>, "-|>"),
    edge(<r3>, <r1>, "-|>", stroke: 0.6pt + red),
  ),
  diagram(
    spacing: 3.4em, node-stroke: 0.6pt, node-shape: circle, node-inset: 3pt,
    node((0, 0), text(9pt, $r_1$), name: <s1>), node((1, 0), text(9pt, $r_2$), name: <s2>),
    node((0.5, 0.9), text(9pt, $r_3$), name: <s3>, fill: rgb("#e3ecf7")),
    edge(<s1>, <s2>, "-|>"), edge(<s2>, <s3>, "-|>"),
  ),
  text(8pt)[overwriting pointers], text(8pt)[union-find],
))[
  Grouping joins subtrees along their pointers. Overwriting one representative's pointer
  with another's, without first following each to its root, can close a loop among
  representatives (red), and the pointer jumping after grouping never ends: a run on 5
  ranks hung this way. A union-find links only roots, so the groups stay trees with one
  root each (shaded).
]

#step([Reading past the list], code: local, focus: (13,), diagram(
  spacing: (2.2em, 3.2em), node-stroke: 0.6pt, node-inset: 4.5pt,
  node((0, 0), text(9pt, $u_1$), name: <c0>), node((1, 0), text(9pt, $u_2$), name: <c1>),
  node((2, 0), text(9pt, $u_3$), name: <c2>),
  node((3, 0), text(8pt, fill: red)[$0$], name: <c3>, stroke: (paint: red, dash: "dashed", thickness: 0.6pt)),
  node((1.5, -0.55), text(7.5pt)[unresolved subtrees], stroke: none),
  node((3, -0.55), text(7.5pt, fill: red)[one past the end], stroke: none),
  node((1.5, 1), text(8pt)[subtree of point 0], name: <z>, fill: rgb("#f6d7d2")),
  edge(<c0>, <z>, "-|>", stroke: 0.6pt + red), edge(<c1>, <z>, "-|>", stroke: 0.6pt + red),
  edge(<c2>, <z>, "-|>", stroke: 0.6pt + red),
  edge(<c3>, <z>, "--|>", stroke: 0.6pt + red, text(7.5pt, fill: red)[read as the target], label-side: left),
))[
  The shared-memory grouping read one element past its list of unresolved subtrees. The
  stray value was usually 0, so it formed no groups and pointed every unresolved subtree
  at the subtree of point 0, merging them into whatever cluster point 0 was in.
  AddressSanitizer reported the read. Only multithreaded runs reach this line, because
  sequential in-place updates collapse cycles on their own.
]

#step([Distances exactly at $epsilon$], code: one-rank, focus: (1,), image("figures/eps-boundary.svg", width: 100%))[
  On a unit grid with $epsilon = 1$ and $M = #ex.eps_grid.min_samples$, each interior point
  has four neighbors at distance exactly $epsilon$. With $<=$, as in @lem:classify and in
  scikit-learn, interior points are core (blue) and the rest border (gold) or noise. The
  strict $<$ of the original implementation finds no core point. Grids and rounded
  distances put many points exactly at $epsilon$.
]

#step([A finite "no edge" weight], code: local, focus: (3,), image("figures/sentinel.svg", width: 100%))[
  The search for minimum edges starts each subtree at a "no edge yet" weight $s$ and keeps
  only lighter edges. With a finite $s$, edges of weight $s$ or more (faint red) can never
  be selected, and clusters fall apart, here into #ex.sentinel_pieces pieces. The original
  $s = 10^4$ did this to data whose neighbor distances reach $10^4$, such as raw physical
  units; here $s$ is the 60th percentile of the example's edge weights.
]

#step([A loop variable shared by threads], code: one-rank, focus: (6,), table(
  columns: 4,
  align: (center, left, left, center),
  table.header[][thread A][thread B][shared $u$],
  [1], [$u <- R[p]$], [], [$R[p]$],
  [2], [], [$u <- R[q]$], [$R[q]$],
  [3], [#text(fill: red)[$R[p] <- "root"(u)$]], [], [$R[q]$],
))[
  The loop that gives each core point its cluster used one variable $u$ for all threads.
  Thread B can overwrite it between thread A's read and A's use, and point $p$ receives
  the cluster of B's point $q$. Declaring $u$ inside the loop makes it private to each
  iteration. Sanitized, slower builds hit this timing; ordinary runs rarely did.
]

#step([One writer for the output file], table(
  columns: 3,
  align: (center, left, left),
  table.header[][rank 0][rank 2 (late)],
  [1], [open, truncate], [_still computing_],
  [2], [write $n$ labels], [],
  [3], [], [#text(fill: red)[open, truncate]],
))[
  Outside the algorithm, in the command-line program: every rank opened, and so emptied,
  the labels file, but only rank 0 wrote it. A rank that reached the open after rank 0
  had finished left an empty file, which happened on oversubscribed CI machines. Only
  rank 0 may open it.
]
