// Stage VI of knndbscan() (Algorithm 1, ClusterBorder): each non-core point within eps
// of a core point among its neighbors joins that core point's cluster, preferring a
// neighbor on its own rank. Part of the unity build of knndbscan.cpp.
#include "globals.h"
using namespace std;

void label_border_points(const point_int n0, const point_int n, const point_int ISTART,
                         vector<point_int>& R, const point_int* JA, const float* A) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // sort borders
    vector<Cycle> borders;
#pragma omp parallel for reduction(merge_Cycle : borders)
    for (point_int p = 0; p < n; p++) {
        edge_int m = p * maxk;
        if ((R[p] == -1) and (A[m] <= eps)) {  // non-core point and not obvious noise
            Cycle e = {p, -1};
            while (A[m] <= eps) {
                point_int q = JA[m] - ISTART;
                if ((q < 0) or (q >= n)) {
                    m += 1;
                } else if (R[q] > -1) {
                    e.j = R[q];
                    break;
                } else {
                    m += 1;
                }
            }
            borders.push_back(e);
        }
    }
    int n_borders = borders.size();

    // sort crossE
    vector<edge_int> crossE;
#pragma omp parallel for reduction(merge_edgeint : crossE)
    for (int l = 0; l < n_borders; l++) {
        Cycle e = borders[l];
        if (e.j == -1) {  // only points need to be checked
            point_int p = e.i;
            edge_int m = p * maxk;
            while (A[m] <= eps) {
                point_int q = JA[m] - ISTART;
                if ((q < 0) or (q >= n)) crossE.push_back(m);
                m += 1;
            }
        }
    }

    // find border points and update their roots from information of crossedges
    vector<point_int> sbuf, rbuf, ordering;
    int* scounts = new int[gsize];
    int* sdispls = new int[gsize];
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    for (int l = 0; l < gsize; l++) scounts[l] = 0;

    int nE = crossE.size();
    int i, j, k, pointer;
    // prepare send
    for (int l = 0; l < nE; l++) {
        j = JA[crossE[l]];
        k = floor((double)j / (double)n0);
        scounts[k] += 1;
    }
    for (int l = 0; l < gsize; l++)
        sdispls[l] = (l == 0) ? 0 : sdispls[l - 1] + scounts[l - 1];
    sbuf.resize(nE);
    ordering.resize(nE);
    vector<int> counts(gsize, 0);
    for (int l = 0; l < nE; l++) {
        j = JA[crossE[l]];
        k = floor((double)j / (double)n0);
        pointer = sdispls[k] + counts[k];
        sbuf[pointer] = j;
        ordering[l] = pointer;
        counts[k] += 1;
    }
    // 1st all to all
    MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, MPI_COMM_WORLD);
    k = 0;
    for (int l = 0; l < gsize; l++) {
        rdispls[l] = (l == 0) ? 0 : rdispls[l - 1] + rcounts[l - 1];
        k += rcounts[l];
    }
    rbuf.resize(k);
    MPI_Alltoallv(sbuf.data(), scounts, sdispls, MPI_INT, rbuf.data(), rcounts, rdispls,
                  MPI_INT, MPI_COMM_WORLD);

    // update and 2nd all to all to send back new pointer
    for (int l = 0; l < k; l++) {
        j = rbuf[l] - ISTART;
        rbuf[l] = R[j];
    }
    MPI_Alltoallv(rbuf.data(), rcounts, rdispls, MPI_INT, sbuf.data(), scounts, sdispls,
                  MPI_INT, MPI_COMM_WORLD);

    // check if border and change R
    edge_int r;
    for (int l = 0; l < nE; l++) {
        r = crossE[l];
        i = floor((double)r / (double)maxk);
        if (R[i] == -1) {  // need to check
            pointer = ordering[l];
            R[i] = sbuf[pointer];
        }
    }

    // update border points from local information: using borders
    Cycle c;
    for (int l = 0; l < n_borders; l++) {
        c = borders[l];
        if (c.j >= 0) R[c.i] = c.j;
    }
}
