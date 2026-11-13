// Single-process implementation of the MPI calls in cpp/core and cpp/cli, used by the
// shared library so that it builds and runs without an MPI installation.
//
// Every communicator has exactly one rank, so each collective copies the calling rank's
// send block into its receive block. A datatype is represented by its extent in bytes.
//
// All functions are static inline on purpose: an exported MPI_* symbol could be
// interposed by a real libmpi loaded into the same process (mpi4py loads it with
// RTLD_GLOBAL), which would hand these fake handles to the real library.
#ifndef KNNDBSCAN_MPI_SERIAL_H
#define KNNDBSCAN_MPI_SERIAL_H

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

typedef int MPI_Comm;
typedef std::ptrdiff_t MPI_Aint;
typedef std::size_t MPI_Datatype;

#define MPI_COMM_WORLD 0
#define MPI_INT (static_cast<MPI_Datatype>(sizeof(int)))
#define MPI_FLOAT (static_cast<MPI_Datatype>(sizeof(float)))
#define MPI_SUCCESS 0
#define MPI_THREAD_FUNNELED 1
#define MPI_THREAD_SERIALIZED 2

static inline void mpi_serial_copy(void* dst, const void* src, std::size_t bytes) {
    // Empty vectors pass &v[0], which may be null; memcpy with a null pointer is UB.
    if (bytes > 0) std::memcpy(dst, src, bytes);
}

static inline int MPI_Init_thread(int*, char***, int required, int* provided) {
    *provided = required;
    return MPI_SUCCESS;
}
static inline int MPI_Finalize() { return MPI_SUCCESS; }
static inline int MPI_Comm_rank(MPI_Comm, int* rank) {
    *rank = 0;
    return MPI_SUCCESS;
}
static inline int MPI_Comm_size(MPI_Comm, int* size) {
    *size = 1;
    return MPI_SUCCESS;
}
static inline int MPI_Barrier(MPI_Comm) { return MPI_SUCCESS; }
static inline double MPI_Wtime() { return omp_get_wtime(); }

// The extent is rounded up to the largest member size, matching the padding the
// compiler gives the C struct the type describes (all members are 4-byte scalars).
static inline int MPI_Type_create_struct(int count, const int* blocklengths,
                                         const MPI_Aint* displacements,
                                         const MPI_Datatype* types,
                                         MPI_Datatype* newtype) {
    std::size_t extent = 0, alignment = 1;
    for (int i = 0; i < count; ++i) {
        extent = std::max(extent, static_cast<std::size_t>(displacements[i]) +
                                      blocklengths[i] * types[i]);
        alignment = std::max(alignment, types[i]);
    }
    *newtype = (extent + alignment - 1) / alignment * alignment;
    return MPI_SUCCESS;
}
static inline int MPI_Type_commit(MPI_Datatype*) { return MPI_SUCCESS; }

static inline int MPI_Allgather(const void* sendbuf, int sendcount,
                                MPI_Datatype sendtype, void* recvbuf, int, MPI_Datatype,
                                MPI_Comm) {
    mpi_serial_copy(recvbuf, sendbuf, sendcount * sendtype);
    return MPI_SUCCESS;
}
static inline int MPI_Allgatherv(const void* sendbuf, int sendcount,
                                 MPI_Datatype sendtype, void* recvbuf, const int*,
                                 const int* displs, MPI_Datatype recvtype, MPI_Comm) {
    mpi_serial_copy(static_cast<char*>(recvbuf) + displs[0] * recvtype, sendbuf,
                    sendcount * sendtype);
    return MPI_SUCCESS;
}
static inline int MPI_Alltoall(const void* sendbuf, int sendcount,
                               MPI_Datatype sendtype, void* recvbuf, int, MPI_Datatype,
                               MPI_Comm) {
    mpi_serial_copy(recvbuf, sendbuf, sendcount * sendtype);
    return MPI_SUCCESS;
}
static inline int MPI_Alltoallv(const void* sendbuf, const int* sendcounts,
                                const int* sdispls, MPI_Datatype sendtype,
                                void* recvbuf, const int*, const int* rdispls,
                                MPI_Datatype recvtype, MPI_Comm) {
    mpi_serial_copy(static_cast<char*>(recvbuf) + rdispls[0] * recvtype,
                    static_cast<const char*>(sendbuf) + sdispls[0] * sendtype,
                    sendcounts[0] * sendtype);
    return MPI_SUCCESS;
}
static inline int MPI_Gatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                              void* recvbuf, const int*, const int* displs,
                              MPI_Datatype recvtype, int, MPI_Comm) {
    mpi_serial_copy(static_cast<char*>(recvbuf) + displs[0] * recvtype, sendbuf,
                    sendcount * sendtype);
    return MPI_SUCCESS;
}
static inline int MPI_Scatterv(const void* sendbuf, const int* sendcounts,
                               const int* displs, MPI_Datatype sendtype, void* recvbuf,
                               int, MPI_Datatype, int, MPI_Comm) {
    mpi_serial_copy(recvbuf, static_cast<const char*>(sendbuf) + displs[0] * sendtype,
                    sendcounts[0] * sendtype);
    return MPI_SUCCESS;
}

#endif
