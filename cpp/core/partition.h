// How points and subtrees are divided among MPI ranks (contiguous blocks, in rank
// order), and the sum over ranks the other stages use.
#ifndef KNNDBSCAN_PARTITION_H
#define KNNDBSCAN_PARTITION_H

#include "globals.h"
using namespace std;

inline int find_parent(const int i) {
    int parent = -1;
    int k = 0;
    while (parent < 0) {
        if ((ilabels[k] <= i) and (i < ilabels[k + 1])) parent = k;
        k += 1;
    }
    return parent;
}

inline int countall(const int n) {
    int n_all = 0;
    vector<int> counts(gsize);

    MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < gsize; i++) n_all += counts[i];
    return n_all;
}

inline void label_subtrees(const int rank, const int n_tree, const vector<int> n_trees,
                           const vector<point_int> C, int& n_start, int& n_ALLtrees,
                           map<point_int, int>& label) {
    n_start = 0;
    for (int l = 0; l < rank; l++) n_start += n_trees[l];
    for (int l = 0; l < n_tree; l++) label[C[l]] = n_start + l;
    n_ALLtrees = 0;
    ilabels.resize(gsize + 1);
    ilabels[0] = 0;
    for (int l = 0; l < gsize; l++) {
        n_ALLtrees += n_trees[l];
        ilabels[l + 1] = n_ALLtrees;
    }
    jlabels.resize(n_ALLtrees);
#pragma omp parallel for
    for (int l = 0; l < n_ALLtrees; l++) {
        jlabels[l] = find_parent(l);
    }
}

inline void count_points(const int rank, const point_int N, point_int& n0, point_int& n,
                         point_int& ISTART) {
    point_int n1, p;

    n0 = ceil((double)N / (double)gsize);
    p = ceil((double)N / (double)n0);
    n1 = N - n0 * (p - 1);
    n = n0;
    ISTART = n0 * rank;
    if (rank == p - 1) {
        n = n1;
    } else if (rank > p - 1) {
        n = 0;
        ISTART = N;
    }
}

#endif
