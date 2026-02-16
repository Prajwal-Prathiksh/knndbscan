#include "../include/globals.h"
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

inline bool cas_2(Edge& e_old, const float w_old, const point_int j_new,
                  const float w_new) {
    if (e_old.w == w_old) {
        e_old = {j_new, w_new};
        return true;  // swap success, and terminate write
    } else {
        return false;  // swap failed, need to check if it is still necessary for swap
    }
}

inline void pwrite_2(Edge& e_old, const point_int j_new, const float w_new) {
    // compare and swap smaller edges
    float w_old;
    do {
        w_old = e_old.w;
    } while ((w_new < w_old) and (cas_2(e_old, w_old, j_new, w_new)));
    // the first condition determines if it is necessary to swap
    // the second condition determines if the swap is successed.
}

void minimum_edges(MPI_Datatype Edgetype, const vector<int> C,
                   const vector<int> C_local, const vector<Edge> minE,
                   map<int, int>& best) {
    int rank, n_start, n_end;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    n_start = ilabels[rank];
    n_end = ilabels[rank + 1];

    // output: best
    int* scounts = new int[gsize];
    int* rcounts = new int[gsize];
    int* sdispls = new int[gsize];
    int* rdispls = new int[gsize];
    for (int l = 0; l < gsize; l++) scounts[l] = 0;

    vector<fullEdge> pre_sbuf, sbuf, redges, rbuf;

    // assgin sbuf
    int c, k;
    Edge edge;
    for (int i = 0; i < C.size(); i++) {
        c = C[i];
        edge = minE[c];
        if (edge.j >= 0) {
            if ((n_start <= c) and (c < n_end)) {  // local
                redges.push_back({c, edge.j, edge.w});
            } else {
                k = jlabels[c];
                scounts[k] += 1;
                pre_sbuf.push_back({c, edge.j, edge.w});
            }
        }
    }
    for (int l = 0; l < gsize; l++)
        sdispls[l] = (l == 0) ? 0 : sdispls[l - 1] + scounts[l - 1];
    int pointer, nout;
    nout = pre_sbuf.size();
    sbuf.resize(nout);
    fullEdge e;
    vector<int> counts(gsize);
    for (int l = 0; l < nout; l++) {
        e = pre_sbuf[l];
        c = e.i;
        k = jlabels[c];
        pointer = counts[k] + sdispls[k];
        sbuf[pointer] = e;
        counts[k] += 1;
    }
    // ALL to all transpose edges
    vector<int> n1(gsize), n2(gsize), s1(gsize), s2(gsize);
#pragma omp parallel for
    for (int i = 0; i < gsize; i++) {
        n1[i] = 1;
        n2[i] = 1;
        s1[i] = i;
        s2[i] = i;
    }

    MPI_Alltoallv(scounts, &n1[0], &s1[0], MPI_INT, rcounts, &n2[0], &s2[0], MPI_INT,
                  MPI_COMM_WORLD);
    // MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, MPI_COMM_WORLD);
    k = 0;
    for (int l = 0; l < gsize; l++) {
        rdispls[l] = (l == 0) ? 0 : rdispls[l - 1] + rcounts[l - 1];
        k += rcounts[l];
    }
    rbuf.resize(k);
    MPI_Alltoallv(&sbuf[0], scounts, sdispls, Edgetype, &rbuf[0], rcounts, rdispls,
                  Edgetype, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // compare and sort
    int n_local = C_local.size();
    for (int i = 0; i < n_local; i++) {
        c = C_local[i];
        best[c] = -1;
    }

    redges.insert(redges.end(), rbuf.begin(), rbuf.end());
    fullEdge redge;
    map<int, float> minW;
    for (int i = 0; i < redges.size(); i++) {
        redge = redges[i];
        c = redge.i;
        if (best[c] == -1) {
            best[c] = redge.j;
            minW[c] = redge.w;
        } else if (redge.w < minW[c]) {
            best[c] = redge.j;
            minW[c] = redge.w;
        }
    }
}

inline void prepare_send(const int nsend, int* sdispls, const map<int, int> best,
                         vector<int>& sbuf, const vector<int> pre_check,
                         vector<int>& to_check) {
    to_check.resize(nsend);
    sbuf.resize(nsend);
    vector<int> counts(gsize, 0);
    // #pragma omp parallel for reduction(vec_plus:counts)
    for (int l = 0; l < nsend; l++) {
        int i = pre_check[l];
        int j = best.at(i);
        int k = jlabels[j];
        int pointer = sdispls[k] + counts[k];
        to_check[pointer] = i;
        sbuf[pointer] = j;
        counts[k] += 1;
    }
}

inline void alltoall_update(int* scounts, int* sdispls, vector<int>& sbuf,
                            const map<int, int> best) {
    vector<int> rbuf;
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];

    // 1st all to all
    vector<int> n1(gsize), n2(gsize), s1(gsize), s2(gsize);
#pragma omp parallel for
    for (int i = 0; i < gsize; i++) {
        n1[i] = 1;
        n2[i] = 1;
        s1[i] = i;
        s2[i] = i;
    }

    MPI_Alltoallv(scounts, &n1[0], &s1[0], MPI_INT, rcounts, &n2[0], &s2[0], MPI_INT,
                  MPI_COMM_WORLD);
    //    MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, MPI_COMM_WORLD);
    int k = 0;
    for (int l = 0; l < gsize; l++) {
        rdispls[l] = (l == 0) ? 0 : rdispls[l - 1] + rcounts[l - 1];
        k += rcounts[l];
    }
    rbuf.resize(k);
    MPI_Alltoallv(&sbuf[0], scounts, sdispls, MPI_INT, &rbuf[0], rcounts, rdispls,
                  MPI_INT, MPI_COMM_WORLD);
// update and 2nd all to all to send back new pointer
#pragma omp parallel for
    for (int l = 0; l < k; l++) {
        int j = rbuf[l];
        rbuf[l] = best.at(j);
    }

    MPI_Alltoallv(&rbuf[0], rcounts, rdispls, MPI_INT, &sbuf[0], scounts, sdispls,
                  MPI_INT, MPI_COMM_WORLD);
}

inline void post_recv(const int n_check, int* scounts, const vector<int> sbuf,
                      const vector<int> to_check, const vector<int> to_compare,
                      vector<int>& pre_check, map<int, int>& best) {
    int i, j, k;
    for (int l = 0; l < gsize; l++) scounts[l] = 0;
    pre_check.resize(0);
    for (int l = 0; l < n_check; l++) {
        i = to_check[l];
        j = sbuf[l];               // new pointer
        best[i] = j;               // update
        if (to_compare[l] != j) {  // root not found, need the next iteration
            k = jlabels[j];
            scounts[k] += 1;
            pre_check.push_back(i);
        }
    }
}

inline int countall(const int n) {
    int n_all = 0;
    vector<int> counts(gsize);

    MPI_Allgather(&n, 1, MPI_INT, &counts[0], 1, MPI_INT, MPI_COMM_WORLD);
    for (int i = 0; i < gsize; i++) n_all += counts[i];
    return n_all;
}

void count_cycles(const vector<int> I, vector<int>& J) {
    // using the similar way as forming the local branches
    map<int, int> R;
    int k = I.size();

    for (int l = 0; l < k; l++) {
        R[I[l]] = -1;
    }

    map<int, int> r_index;
    int i, j, r, s;
    int n_local = 0;

    for (int l = 0; l < k; l++) {
        i = I[l];
        j = J[l];
        if (R[i] < 0) {
            if (R[j] < 0) {
                R[i] = n_local;
                R[j] = n_local;
                r_index[n_local] = n_local;
                n_local += 1;
            } else {
                R[i] = R[j];
            }
        } else {
            if (R[j] < 0) {
                R[j] = R[i];
            } else {
                r = R[i];
                s = R[j];
                if (r_index[r] != r_index[s]) {
                    r_index[r] = r_index[s];
                    R[i] = s;
                }
            }
        }
    }

    // break symmetry
    for (int l = 0; l < r_index.size(); l++) {
        r = r_index[l];
        s = r_index[r];
        if ((s == l) and (l < r)) r_index[l] = l;
    }

    int n_cycles = 0;
    map<int, point_int> roots;
    for (int l = 0; l < n_local; l++) {
        if (r_index[l] == l) {
            n_cycles += 1;
            roots[l] = -1;
        } else {
            while (r_index[r_index[l]] != r_index[l]) {
                r_index[l] = r_index[r_index[l]];
            }
        }
    }
    cout << "number of points and cycles:" << k << " " << n_cycles << endl;

    // update roots
    int c;
    for (int l = 0; l < k; l++) {
        i = I[l];
        c = r_index[R[i]];
        if (roots[c] == -1) {
            roots[c] = i;
            J[l] = i;
        } else {
            J[l] = roots[c];
        }
    }
}

void break_cycles_global(const vector<int> pre_check, vector<int>& roots) {
    //
    int n_check, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    n_check = pre_check.size();
    vector<int> counts(gsize);
    MPI_Allgather(&n_check, 1, MPI_INT, &counts[0], 1, MPI_INT, MPI_COMM_WORLD);
    int n_all = 0;
    for (int i = 0; i < gsize; i++) n_all += counts[i];
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    vector<int> I1(n_all), J1(n_all);
    if (rank == 0) {
        for (int i = 0; i < gsize; i++) {
            rcounts[i] = counts[i];
            rdispls[i] = (i == 0) ? 0 : rdispls[i - 1] + rcounts[i - 1];
        }
    }
    MPI_Gatherv(&pre_check[0], n_check, MPI_INT, &I1[0], rcounts, rdispls, MPI_INT, 0,
                MPI_COMM_WORLD);
    MPI_Gatherv(&roots[0], n_check, MPI_INT, &J1[0], rcounts, rdispls, MPI_INT, 0,
                MPI_COMM_WORLD);

    // break cycles
    if (rank == 0) {
        count_cycles(I1, J1);
    }

    // distribute new roots
    MPI_Scatterv(&J1[0], rcounts, rdispls, MPI_INT, &roots[0], n_check, MPI_INT, 0,
                 MPI_COMM_WORLD);
}

void output_best(const map<int, int> best, const string name) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int istart = ilabels[rank];
    int iend = ilabels[rank + 1];
    int n_all = ilabels[gsize];
    int nbuf = iend - istart;
    vector<int> sbuf(nbuf);
    for (int i = 0; i < nbuf; i++) sbuf[i] = best.at(istart + i);

    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    if (rank == 0) {
        for (int i = 0; i < gsize; i++) {
            rcounts[i] = ilabels[i + 1] - ilabels[i];
            rdispls[i] = (i == 0) ? 0 : rdispls[i - 1] + rcounts[i - 1];
        }
    }

    vector<int> rbuf;
    if (rank == 0) rbuf.resize(n_all);
    MPI_Gatherv(&sbuf[0], nbuf, MPI_INT, &rbuf[0], rcounts, rdispls, MPI_INT, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        ofstream fout;
        fout.open(name);
        for (int i = 0; i < n_all; i++) {
            fout << i << "  " << rbuf[i] << "\n";
        }
        fout.close();
    }
}

void find_roots(const vector<int> C_local, map<int, int>& best) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int* scounts = new int[gsize];
    int* sdispls = new int[gsize];
    for (int l = 0; l < gsize; l++) scounts[l] = 0;

    vector<int> pre_check, to_check, sbuf, to_compare;

    int check_flag, n_check, i, j, k;
    // I: break symmetry
    int n_local = C_local.size();
    // search local roots and assign pre_check
    for (int l = 0; l < n_local; l++) {
        i = C_local[l];
        if (best[i] < 0) {
            best[i] = i;  // a local root with no output edge
        } else {
            pre_check.push_back(i);
            j = best[i];
            k = jlabels[j];
            scounts[k] += 1;
        }
    }
    n_check = pre_check.size();

    // assign: to_check, sbuf
    for (int l = 0; l < gsize; l++)
        sdispls[l] = (l == 0) ? 0 : sdispls[l - 1] + scounts[l - 1];
    prepare_send(n_check, sdispls, best, sbuf, pre_check, to_check);
    // update pointer
    alltoall_update(scounts, sdispls, sbuf, best);
    // find local roots and break symmetry
    for (int l = 0; l < gsize; l++) scounts[l] = 0;
    pre_check.resize(0);
    for (int l = 0; l < n_check; l++) {
        i = to_check[l];
        j = sbuf[l];   // new pointer
        if (i != j) {  // root not found
            k = jlabels[j];
            scounts[k] += 1;
            pre_check.push_back(i);
            best[i] = j;
        } else if (i < best[i]) {
            best[i] = i;  // a local root that is connected to another subtree
        }
    }
    n_check = pre_check.size();
    check_flag = countall(n_check);
    // II: pointer jumping
    map<int, int> best_copy(best);
    int check_flag_old = -1;
    int istep = 0;
    while (check_flag > 0) {
        // assign: to_check, sbuf
        for (int l = 0; l < gsize; l++)
            sdispls[l] = (l == 0) ? 0 : sdispls[l - 1] + scounts[l - 1];
        prepare_send(n_check, sdispls, best, sbuf, pre_check, to_check);
        to_compare = sbuf;
        // all to all transpose to update pointer
        alltoall_update(scounts, sdispls, sbuf, best);
        // update best and pre_check for next iteration
        post_recv(n_check, scounts, sbuf, to_check, to_compare, pre_check, best);
        n_check = pre_check.size();
        check_flag = countall(n_check);
        if ((check_flag != 0) and (check_flag == check_flag_old)) {
            vector<int> roots(n_check);
            for (int l = 0; l < n_check; l++) {
                int c = pre_check[l];
                roots[l] = best_copy.at(c);
            }
            break_cycles_global(pre_check, roots);
            for (int l = 0; l < n_check; l++) {
                int c = pre_check[l];
                best[c] = roots[l];
            }
            check_flag = 0;
        }
        check_flag_old = check_flag;
    }
}

int update_roots(const map<int, int> best, vector<int>& C, vector<int>& C_local,
                 vector<int>& roots, vector<edge_int>& crossE) {
    int rank, n_start, n_end;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    n_start = ilabels[rank];
    n_end = ilabels[rank + 1];

    vector<int> sbuf, ordering;
    int i, j, k, pointer;
    int* scounts = new int[gsize];
    int* sdispls = new int[gsize];
    for (int l = 0; l < gsize; l++) scounts[l] = 0;

    // 1. update roots
    map<int, int> C_root, new_C;
    int n_C = C.size();
    sbuf.resize(n_C);
    ordering.resize(n_C);
    for (int l = 0; l < n_C; l++) {
        k = jlabels[C[l]];
        scounts[k] += 1;
    }
    vector<int> counts(gsize, 0);
    for (int l = 0; l < gsize; l++)
        sdispls[l] = (l == 0) ? 0 : sdispls[l - 1] + scounts[l - 1];
    for (int l = 0; l < n_C; l++) {
        i = C[l];
        k = jlabels[i];
        pointer = sdispls[k] + counts[k];
        sbuf[pointer] = i;
        ordering[l] = pointer;
        counts[k] += 1;
    }
    alltoall_update(scounts, sdispls, sbuf, best);
    // obtain: C_root, new_C
    for (int l = 0; l < n_C; l++) {
        i = C[l];
        pointer = ordering[l];
        j = sbuf[pointer];
        C_root[i] = j;
        new_C[j] = 1;
    }

    // update C and C_local
    map<int, int>::iterator itr;
    C.resize(0);
    C_local.resize(0);
    for (itr = new_C.begin(); itr != new_C.end(); ++itr) {
        i = itr->first;
        C.push_back(i);
        if ((n_start <= i) and (i < n_end)) C_local.push_back(i);
    }

    // update roots
    for (int l = 0; l < roots.size(); l++) {
        i = roots[l];
        roots[l] = C_root[i];
    }
    // recover roots
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    int n_root = roots.size();
    fill(counts.begin(), counts.end(), 0);
    MPI_Allgather(&n_root, 1, MPI_INT, &counts[0], 1, MPI_INT, MPI_COMM_WORLD);
    int n_allroot = 0;
    for (int l = 0; l < gsize; l++) {
        n_allroot += counts[l];
        rcounts[l] = counts[l];
        rdispls[l] = (l == 0) ? 0 : rdispls[l - 1] + rcounts[l - 1];
    }

    vector<int> all_roots(n_allroot);
    MPI_Allgatherv(&roots[0], n_root, MPI_INT, &all_roots[0], rcounts, rdispls, MPI_INT,
                   MPI_COMM_WORLD);
    // update edges
    int nE = crossE.size();
#pragma omp parallel for
    for (int l = 0; l < nE; l++) {
        if (crossE[l] >= 0) {
            int i = all_roots[II[l]];
            int j = all_roots[JJ[l]];
            if (i != j) {
                II[l] = i;
                JJ[l] = j;
            } else {
                crossE[l] = (edge_int)-1;
            }
        }
    }
    n_C = C_local.size();
    int n_tree = countall(n_C);
    return n_tree;
}

void relabel(const vector<int> C_local, vector<int>& roots) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // relabel
    int n = C_local.size();
    vector<int> counts(gsize, 0);
    MPI_Allgather(&n, 1, MPI_INT, &counts[0], 1, MPI_INT, MPI_COMM_WORLD);

    int count = 0;
    int* rcounts = new int[gsize];
    int* rdispls = new int[gsize];
    rdispls[0] = 0;
    for (int l = 0; l < gsize; l++) {
        rcounts[l] = counts[l];
        count += counts[l];
        if (l != 0) rdispls[l] = rdispls[l - 1] + rcounts[l - 1];
    }
    vector<int> rbuf(count);
    MPI_Allgatherv(&C_local[0], n, MPI_INT, &rbuf[0], rcounts, rdispls, MPI_INT,
                   MPI_COMM_WORLD);
    map<int, int> newlabel;
    int k = 0;
    for (int l = 0; l < count; l++) {
        newlabel[rbuf[l]] = k;  // reordering
        k += 1;
    }

    // update roots
    int i;
    for (int l = 0; l < roots.size(); l++) {
        i = newlabel[roots[l]];
        roots[l] = i;
    }
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

void label_borders(const point_int n0, const point_int n, const point_int ISTART,
                   vector<point_int>& R, const point_int* JA, const float* A) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // sort borders
    vector<Cycle> borders;
#pragma omp parallel for reduction(merge_Cycle : borders)
    for (point_int p = 0; p < n; p++) {
        edge_int m = p * maxk;
        if ((R[p] == -1) and (A[m] < eps)) {  // non-core point and not obvious noise
            Cycle e = {p, -1};
            while (A[m] < eps) {
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
            while (A[m] < eps) {
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
    MPI_Alltoallv(&sbuf[0], scounts, sdispls, MPI_INT, &rbuf[0], rcounts, rdispls,
                  MPI_INT, MPI_COMM_WORLD);

    // update and 2nd all to all to send back new pointer
    for (int l = 0; l < k; l++) {
        j = rbuf[l] - ISTART;
        rbuf[l] = R[j];
    }
    MPI_Alltoallv(&rbuf[0], rcounts, rdispls, MPI_INT, &sbuf[0], scounts, sdispls,
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

vector<int> cutmst(const int n_ALLtrees, int& n_local, vector<edge_int>& crossE,
                   const float* A, bool verbose = false) {
    // n_ALLtrees: total number of subtrees over all processes
    // n_local: # of subtress in local process

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // declare new MPI type: Edgetype
    MPI_Datatype Edgetype;
    MPI_Datatype type[3] = {MPI_INT, MPI_INT, MPI_FLOAT};
    int blocklen[3] = {1, 1, 1};
    MPI_Aint disp[3];
    disp[0] = offsetof(struct fullEdge, i);
    disp[1] = offsetof(struct fullEdge, j);
    disp[2] = offsetof(struct fullEdge, w);
    MPI_Type_create_struct(3, blocklen, disp, type, &Edgetype);
    MPI_Type_commit(&Edgetype);

    int nE = crossE.size();
    // Initialization:
    // C: all roots involved with local process.
    // C_local: a subset of C with roots onwed my local process := {i\inC: n_start< i <
    // n_end}
    int n_tree = n_ALLtrees;
    int dn_tree = 1;
    int n_tree_old;

    vector<int> C, C_local, roots;
    int n_start = ilabels[rank];
    for (int l = 0; l < n_local; l++) C.push_back(n_start + l);
    roots = C;
    C_local = C;

    vector<Edge> minE(n_ALLtrees);
    map<int, int> best;

    int istep = 0;
    float w;
    // Boruvka's iterations:
    if (rank == 0 && verbose) cout << "input subtree #:" << n_ALLtrees << endl;
    while (dn_tree > 0) {
        // step 1: initialization
        best.clear();  // \forall i\ in C_local
#pragma omp parallel for
        for (int c = 0; c < n_local; c++) {
            int i = C[c];
            minE[i] = {-1, sentinel};
        }

// step 2-1: select min edges locally: get minE
#pragma omp parallel for
        for (int l = 0; l < nE; l++) {
            edge_int e = crossE[l];
            if (e >= 0) {
                int i = II[l];
                pwrite_2(minE[i], JJ[l], A[e]);
            }
        }

        // step 2-2: send min edges and sort globals min edges: get best
        if (istep > 0) {
            minimum_edges(Edgetype, C, C_local, minE, best);
        } else {
            for (int c = 0; c < n_local; c++) {
                int i = C[c];
                best[i] = minE[i].j;
            }
        }

        // step 3: find roots: update best until roots found
        find_roots(C_local, best);

        // step 4: update for next ieteration: C, C_local, roots, crossE;
        n_tree_old = n_tree;
        n_tree = update_roots(best, C, C_local, roots, crossE);
        int dn_tree_local = n_tree_old - n_tree;
        dn_tree = countall(dn_tree_local);
        n_local = C.size();
        istep += 1;
    }
    if (rank == 0 && verbose) cout << "number of clusters: " << n_tree << endl;

    // relabel
    relabel(C_local, roots);

    return roots;
}
