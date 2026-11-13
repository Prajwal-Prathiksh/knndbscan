// Grouping of the subtrees that pointer jumping leaves without a root, shared by the
// local (local_mst.cpp: break_cycles) and distributed (cut_mst.cpp: count_cycles)
// Boruvka steps.
#ifndef KNNDBSCAN_CYCLES_H
#define KNNDBSCAN_CYCLES_H

#include <map>
#include <vector>

// Each unresolved subtree from[l] points to to[l] through its minimum edge; the
// pointers form cycles with subtrees leading into them. Returns, for every l, the first
// subtree in `from` of the weakly connected group containing from[l], which becomes the
// group's root. A union-find is required: merging groups by overwriting one
// representative's pointer, without first finding its root, can chain representatives
// into a cycle of three or more.
inline std::vector<int> first_member_of_group(const std::vector<int>& from,
                                              const std::vector<int>& to) {
    std::map<int, int> parent;
    for (size_t l = 0; l < from.size(); l++) {
        parent.emplace(from[l], from[l]);
        parent.emplace(to[l], to[l]);
    }
    auto find = [&parent](int x) {
        int root = x;
        while (parent[root] != root) root = parent[root];
        while (parent[x] != root) {
            const int next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    };
    for (size_t l = 0; l < from.size(); l++) parent[find(from[l])] = find(to[l]);

    std::map<int, int> group_root;
    for (size_t l = 0; l < from.size(); l++) group_root.emplace(find(from[l]), from[l]);
    std::vector<int> roots(from.size());
    for (size_t l = 0; l < from.size(); l++) roots[l] = group_root[find(from[l])];
    return roots;
}

#endif
