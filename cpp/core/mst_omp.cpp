#include "cycles.h"
#include "globals.h"
#include "log.h"
using namespace std;

// Helpers of the local Boruvka step (localmst_omp.cpp): lock-free minimum-edge updates
// and cycle breaking.
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
    } while ((w_new < w_old) and (cas(e_old, w_old, j_new, w_new)));
    // the first condition determines if it is necessary to swap
    // the second condition determines if the swap is successed.
}

// Points every subtree left in a cycle of minimum edges (with the subtrees leading into
// it) at the first member of its group, which becomes the group's root.
inline void break_cycles(const vector<Cycle> cycles, vector<Edge>& minE) {
    vector<int> from(cycles.size()), to(cycles.size());
    for (size_t l = 0; l < cycles.size(); l++) {
        from[l] = cycles[l].i;
        to[l] = cycles[l].j;
    }
    const vector<int> roots = first_member_of_group(from, to);
    for (size_t l = 0; l < cycles.size(); l++) minE[from[l]].j = roots[l];
    KNN_DEBUG_EVERY_RANK("local MST: joined %zu subtree(s) caught in cycles",
                         cycles.size());
}
