#include "cut_edges.cpp"
#include "cut_mst.cpp"
#include "globals.h"
#include "local_mst.cpp"
#include "log.h"
using namespace std;

int gsize = 0;
int num_threads = 0;
float eps = 0.0;
// Weight of "no minimum edge yet": must exceed every edge weight, so it is infinite.
float sentinel = numeric_limits<float>::infinity();
int minPts = 0;
int maxk = 0;  // nearest neighbor # stored in graph for each point, may not be the same
               // as minPts
vector<int> ilabels;
vector<int> jlabels;
vector<int> II;
vector<int> JJ;

vector<point_int> knndbscan(const point_int N, const float eps_value,
                            const int minPts_value, const int maxk_value,
                            const point_int* JA, const float* A) {
    // initialize basic parameters
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    gsize = nprocs;
    minPts = minPts_value;
    maxk = maxk_value;
    eps = eps_value;

#pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    point_int n0, n, ISTART;
    count_points(rank, N, n0, n, ISTART);
    knnlog::rank = rank;
    knnlog::n_ranks = nprocs;
    KNN_INFO("clustering %d points: eps=%g, min_samples=%d, %d rank(s) x %d thread(s)",
             N, eps, minPts, gsize, num_threads);
    KNN_DEBUG_EVERY_RANK("owns points [%d, %d)", ISTART, ISTART + n);

    double t0 = MPI_Wtime();

    // I: determine core/non-core points
    vector<point_int> R(n);
    vector<point_int> core;
#pragma omp parallel for reduction(merge_pointint : core)
    for (point_int p = 0; p < n; p++) {
        float radius = A[p * maxk + minPts - 2];
        if (radius <= eps) {
            R[p] = p;  // mark as core
            core.push_back(p);
        } else {
            R[p] = (point_int)-1;  // mark as non-core
        }
    }
    point_int n_core = core.size();
    int n_ALLcore = countall(n_core);

    KNN_INFO("core points: %d of %d", n_ALLcore, N);

    // II: construct local mst and select crossE: C, R, crossE
    int n_tree;
    vector<point_int> C = core;
    vector<edge_int> crossE;
    n_tree = localmst_omp(n, ISTART, JA, A, R, C, core, crossE);
    int nE = crossE.size();
    MPI_Barrier(MPI_COMM_WORLD);
    double t_local = MPI_Wtime();

    KNN_DEBUG_EVERY_RANK("local MST: %d subtree(s), %d cut edge(s)", n_tree, nE);

    // cunt cross edges #s:
    vector<int> nEs(gsize);
    MPI_Allgather(&nE, 1, MPI_INT, nEs.data(), 1, MPI_INT, MPI_COMM_WORLD);
    edge_int total_nE = nEs[0];
    edge_int max_nE = nEs[0];
    edge_int min_nE = nEs[0];

    for (int l = 1; l < gsize; l++) {
        if (nEs[l] < min_nE) min_nE = nEs[l];
        if (nEs[l] > max_nE) max_nE = nEs[l];
        total_nE += nEs[l];
    }
    double avg_nE = (double)total_nE / (double)gsize;

    // III: relabel local roots: label, ilabels;
    vector<point_int> n_trees(gsize);
    map<point_int, int> label;
    MPI_Allgather(&n_tree, 1, MPI_INT, n_trees.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int n_start, n_ALLtrees;
    label_subtrees(rank, n_tree, n_trees, C, n_start, n_ALLtrees, label);
    MPI_Barrier(MPI_COMM_WORLD);
    KNN_INFO("local MST: %d subtree(s) in %.3e s", n_ALLtrees, t_local - t0);
    KNN_DEBUG("cut edges per rank: min %lld, max %lld, mean %.1f", (long long)min_nE,
              (long long)max_nE, avg_nE);

    // IV: update cut edges
    update_cutedges(n0, ISTART, R, label, JA, A, crossE);
    MPI_Barrier(MPI_COMM_WORLD);

    double t_cutE = MPI_Wtime();
    KNN_INFO("cut edges: %lld, relabelled in %.3e s", (long long)total_nE,
             t_cutE - t_local);

    // V: construct cut mst: R
    vector<int> roots;
    roots = cutmst(n_ALLtrees, n_tree, crossE, A);

    map<int, int>::iterator itr;
    point_int u;
    int v;
    for (itr = label.begin(); itr != label.end(); ++itr) {
        u = itr->first;
        v = itr->second;
        label[u] = roots[v - n_start];
    }
    // root is private to each iteration and the map is only read: a variable shared by
    // the threads here let one thread label its point with another thread's root.
#pragma omp parallel for
    for (point_int p = 0; p < n; p++) {
        const point_int root = R[p];
        if (root != -1) R[p] = label.at(root);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double t_cutmst = MPI_Wtime();
    KNN_INFO("cut MST in %.3e s", t_cutmst - t_cutE);

    // VI: mark border points
    label_borders(n0, n, ISTART, R, JA, A);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    KNN_INFO("border points in %.3e s; clustering took %.3e s", t_end - t_cutmst,
             t_end - t0);
    return R;
}
