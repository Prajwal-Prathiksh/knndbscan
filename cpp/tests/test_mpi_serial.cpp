// Checks cpp/mpi_serial.h against a one-rank MPI communicator, the semantics cpp/core
// relies on: each collective copies the send block, at its displacement, to the receive
// block at its displacement, and a struct datatype has the extent of the C struct it
// describes.
#include <cstdio>
#include <vector>

#include "globals.h"  // Edge, fullEdge; includes mpi_serial.h without KNNDBSCAN_WITH_MPI

static int failures = 0;

#define CHECK(condition)                                                          \
    do {                                                                          \
        if (!(condition)) {                                                       \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, \
                         #condition);                                             \
            failures++;                                                           \
        }                                                                         \
    } while (0)

static void test_world_has_one_rank() {
    int rank = -1, size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    CHECK(rank == 0 && size == 1);
}

// cutmst.cpp sends fullEdge with exactly this construction; a member added to the
// struct without updating it would make the extent and sizeof disagree.
static void test_struct_types_have_the_extent_of_their_structs() {
    MPI_Datatype full_type, edge_type;
    MPI_Datatype full_types[3] = {MPI_INT, MPI_INT, MPI_FLOAT};
    int full_lengths[3] = {1, 1, 1};
    MPI_Aint full_displacements[3] = {offsetof(fullEdge, i), offsetof(fullEdge, j),
                                      offsetof(fullEdge, w)};
    MPI_Type_create_struct(3, full_lengths, full_displacements, full_types, &full_type);
    CHECK(full_type == sizeof(fullEdge));

    MPI_Datatype edge_types[2] = {MPI_INT, MPI_FLOAT};
    int edge_lengths[2] = {1, 1};
    MPI_Aint edge_displacements[2] = {offsetof(Edge, j), offsetof(Edge, w)};
    MPI_Type_create_struct(2, edge_lengths, edge_displacements, edge_types, &edge_type);
    CHECK(edge_type == sizeof(Edge));
}

static void test_collectives_copy_between_displacements() {
    const std::vector<int> send = {9, 1, 2, 3, 9};
    const int one = 1, three = 3, send_at = 1, receive_at = 2;
    const std::vector<int> expected = {0, 0, 1, 2, 3, 0};

    std::vector<int> receive(6, 0);
    MPI_Alltoallv(send.data(), &three, &send_at, MPI_INT, receive.data(), &three,
                  &receive_at, MPI_INT, MPI_COMM_WORLD);
    CHECK(receive == expected);

    receive.assign(6, 0);
    MPI_Allgatherv(send.data() + 1, 3, MPI_INT, receive.data(), &three, &receive_at,
                   MPI_INT, MPI_COMM_WORLD);
    CHECK(receive == expected);

    receive.assign(6, 0);
    MPI_Gatherv(send.data() + 1, 3, MPI_INT, receive.data(), &three, &receive_at,
                MPI_INT, 0, MPI_COMM_WORLD);
    CHECK(receive == expected);

    receive.assign(6, 0);
    MPI_Scatterv(send.data(), &three, &send_at, MPI_INT, receive.data() + 2, 3, MPI_INT,
                 0, MPI_COMM_WORLD);
    CHECK(receive == expected);

    int single = 0;
    MPI_Allgather(send.data() + 2, 1, MPI_INT, &single, 1, MPI_INT, MPI_COMM_WORLD);
    CHECK(single == 2);
    MPI_Alltoall(send.data() + 3, one, MPI_INT, &single, one, MPI_INT, MPI_COMM_WORLD);
    CHECK(single == 3);
}

static void test_struct_collectives_copy_whole_structs() {
    MPI_Datatype full_type;
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_FLOAT};
    int lengths[3] = {1, 1, 1};
    MPI_Aint displacements[3] = {offsetof(fullEdge, i), offsetof(fullEdge, j),
                                 offsetof(fullEdge, w)};
    MPI_Type_create_struct(3, lengths, displacements, types, &full_type);
    const std::vector<fullEdge> send = {{1, 2, 0.5f}, {3, 4, 0.25f}};
    std::vector<fullEdge> receive(2);
    const int two = 2, zero = 0;
    MPI_Alltoallv(send.data(), &two, &zero, full_type, receive.data(), &two, &zero,
                  full_type, MPI_COMM_WORLD);
    CHECK(receive[1].i == 3 && receive[1].j == 4 && receive[1].w == 0.25f);
}

static void test_empty_buffers_are_not_touched() {
    // The core passes data() of empty vectors, which may be null, with zero counts.
    const int zero = 0;
    MPI_Alltoallv(nullptr, &zero, &zero, MPI_INT, nullptr, &zero, &zero, MPI_INT,
                  MPI_COMM_WORLD);
    MPI_Gatherv(nullptr, 0, MPI_INT, nullptr, &zero, &zero, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(nullptr, &zero, &zero, MPI_INT, nullptr, 0, MPI_INT, 0,
                 MPI_COMM_WORLD);
    CHECK(true);
}

int main() {
    test_world_has_one_rank();
    test_struct_types_have_the_extent_of_their_structs();
    test_collectives_copy_between_displacements();
    test_struct_collectives_copy_whole_structs();
    test_empty_buffers_are_not_touched();
    if (failures) std::fprintf(stderr, "%d check(s) failed\n", failures);
    return failures != 0;
}
