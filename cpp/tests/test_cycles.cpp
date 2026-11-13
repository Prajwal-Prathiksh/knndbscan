// Checks first_member_of_group (cpp/core/cycles.h), the grouping both cycle breakers
// use: every subtree must end up pointing at the first member of its weakly connected
// group.
#include <cstdio>
#include <vector>

#include "cycles.h"

static int failures = 0;

static void expect_roots(const std::vector<int>& from, const std::vector<int>& to,
                         const std::vector<int>& expected, const char* name) {
    if (first_member_of_group(from, to) != expected) {
        std::fprintf(stderr, "%s: wrong roots\n", name);
        failures++;
    }
}

int main() {
    // The directed 4-cycle of Fig. 4 of the paper (a power-of-two cycle).
    expect_roots({86, 181, 278, 152}, {181, 278, 152, 86}, {86, 86, 86, 86}, "4-cycle");
    // Two disjoint cycles, one with a subtree (5) leading into it.
    expect_roots({1, 2, 3, 10, 11, 5}, {2, 3, 1, 11, 10, 3}, {1, 1, 1, 10, 10, 1},
                 "two cycles and a tail");
    // Pairs first, then links between the pairs: merging by representative pointers
    // alone can link the representatives of {0, 1}, {2, 3} and {4, 5} into a cycle.
    expect_roots({0, 2, 4, 1, 3, 5}, {1, 3, 5, 2, 4, 0}, {0, 0, 0, 0, 0, 0},
                 "pairs linked in a ring");
    // A pointer to a subtree outside the list still joins the two that share it.
    expect_roots({7, 8}, {99, 99}, {7, 7}, "shared outside target");
    expect_roots({}, {}, {}, "empty");
    if (failures) std::fprintf(stderr, "%d case(s) failed\n", failures);
    return failures != 0;
}
