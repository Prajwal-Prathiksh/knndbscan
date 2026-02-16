#include "cutedges.cpp"
#include "cutmst.cpp"
#include "globals.h"
#include "localmst_omp.cpp"
using namespace std;

int gsize = 0;
int num_threads = 0;
float eps = 0.0;
float sentinel = 10000.;
int minPts = 0;
int maxk = 0;  // nearest neighbor # stored in graph for each point, may not be the same
               // as minPts
vector<int> ilabels;
vector<int> jlabels;
vector<int> II;
vector<int> JJ;

vector<point_int> knndbscan(const point_int N, const float eps_value,
                            const int minPts_value, const int maxk_value,
                            const point_int* JA, const float* A, bool verbose) {
    // initialize basic parameters
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    gsize = nprocs;
    minPts = minPts_value;
    maxk = maxk_value;
    eps = eps_value;
    float sentinel = 10000.;

#pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    point_int n0, n, ISTART;
    count_points(rank, N, n0, n, ISTART);

    double t0 = MPI_Wtime();

    // I: determine core/non-core points
    vector<point_int> R(n);
    vector<point_int> core;
#pragma omp parallel for reduction(merge_pointint : core)
    for (point_int p = 0; p < n; p++) {
        float radius = A[p * maxk + minPts - 2];
        if (radius < eps) {
            R[p] = p;  // mark as core
            core.push_back(p);
        } else {
            R[p] = (point_int)-1;  // mark as non-core
        }
    }
    point_int n_core = core.size();
    int n_ALLcore = countall(n_core);

    if (rank == 0 && verbose)
        cout << "\ntotal number of core points:" << n_ALLcore << " (eps= " << eps
             << ", minPts=" << minPts << ")" << endl;

    // II: construct local mst and select crossE: C, R, crossE
    int n_tree;
    vector<point_int> C = core;
    vector<edge_int> crossE;
    n_tree = localmst_omp(n, ISTART, JA, A, R, C, core, crossE);
    int nE = crossE.size();
    MPI_Barrier(MPI_COMM_WORLD);
    double t_local = MPI_Wtime();

    if (rank == 0 && verbose)
        printf("local mst construction time: %.3E\n", t_local - t0);

    // cunt cross edges #s:
    vector<int> nEs(gsize);
    MPI_Allgather(&nE, 1, MPI_INT, &nEs[0], 1, MPI_INT, MPI_COMM_WORLD);
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
    MPI_Allgather(&n_tree, 1, MPI_INT, &n_trees[0], 1, MPI_INT, MPI_COMM_WORLD);
    int n_start, n_ALLtrees;
    label_subtrees(rank, n_tree, n_trees, C, n_start, n_ALLtrees, label);
    MPI_Barrier(MPI_COMM_WORLD);

    // IV: update cut edges
    update_cutedges(n0, ISTART, R, label, JA, A, crossE);
    MPI_Barrier(MPI_COMM_WORLD);

    double t_cutE = MPI_Wtime();
    if (rank == 0 && verbose) printf("update cut edges time: %.3E\n", t_cutE - t_local);

    // V: construct cut mst: R
    vector<int> roots;
    roots = cutmst(n_ALLtrees, n_tree, crossE, A, verbose);

    map<int, int>::iterator itr;
    point_int u;
    int v;
    for (itr = label.begin(); itr != label.end(); ++itr) {
        u = itr->first;
        v = itr->second;
        label[u] = roots[v - n_start];
    }
#pragma omp parallel for
    for (point_int p = 0; p < n; p++) {
        u = R[p];
        if (u != -1) R[p] = label[u];
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double t_cutmst = MPI_Wtime();
    if (rank == 0 && verbose)
        printf("construct cut mst time: %.3E\n", t_cutmst - t_cutE);

    // VI: mark border points
    label_borders(n0, n, ISTART, R, JA, A);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    // summary
    if (rank == 0 && verbose) {
        printf("total clustering time: %.3E\n", t_end - t0);
        printf("*** cut edges number (max, min, average) : %.3E   %.3E   %.3E\n\n",
               (double)max_nE, (double)min_nE, avg_nE);
    }
    return R;
}
