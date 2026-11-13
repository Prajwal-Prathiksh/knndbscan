// Minimum-edge update of a subtree (Algorithm 3's PWRITE and LOCKFREECAS), shared by
// the local (local_mst.cpp) and the distributed (cut_mst.cpp) Boruvka steps.
//
// The compare-and-swap is a plain compare and assignment, not an atomic operation, so
// threads racing on one subtree may leave an edge that is not the lightest. Clustering
// stays correct: any recorded edge joins the subtree to a neighbor in its connected
// component, which is all the clusters depend on; the spanning tree need not be
// minimal. ThreadSanitizer reports this race.
#ifndef KNNDBSCAN_MIN_EDGES_H
#define KNNDBSCAN_MIN_EDGES_H

#include "globals.h"

inline bool cas(Edge& e_old, const float w_old, const point_int j_new,
                const float w_new) {
    if (e_old.w == w_old) {
        e_old = {j_new, w_new};
        return true;  // swap success, and terminate write
    } else {
        return false;  // swap failed, need to check if it is still necessary for swap
    }
}

inline void pwrite(Edge& e_old, const point_int j_new, const float w_new) {
    // compare and swap smaller edges
    float w_old;
    do {
        w_old = e_old.w;
    } while ((w_new < w_old) && (cas(e_old, w_old, j_new, w_new)));
    // the first condition determines if it is necessary to swap
    // the second condition determines if the swap is successed.
}

#endif
