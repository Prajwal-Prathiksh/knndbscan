#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <mpi.h>
#include <omp.h>

#include "globals.h"
#include "clusters.h"

namespace py = pybind11;

py::array_t<int> knndbscan_py(int N, float eps, int minPts, int k, py::array_t<int> JA_np, py::array_t<float> A_np, int threads = 1, bool verbose = false)
{
    // Initialize MPI if not already initialized
    int initialized;
    MPI_Initialized(&initialized);

    int provided;
    if (!initialized)
    {
        // Request MPI_THREAD_SERIALIZED to allow multiple threads to make MPI calls
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    }

    omp_set_dynamic(0);           // Disable dynamic teams
    omp_set_num_threads(threads); // Set number of threads for OpenMP

    // Get buffer info from numpy arrays
    auto JA_buf = JA_np.request();
    auto A_buf = A_np.request();

    // Zero-copy access to the data
    point_int *JA = static_cast<point_int *>(JA_buf.ptr);
    float *A = static_cast<float *>(A_buf.ptr);

    // Call the clustering function
    std::vector<point_int> labels = knndbscan(N, eps, minPts, k, JA, A, verbose);

    // Don't finalize MPI here - let it persist for multiple calls
    // MPI will be finalized when the Python process exits

    // Return the labels as a numpy array
    return py::array_t<int>(labels.size(), labels.data());
}

PYBIND11_MODULE(_core, m)
{
    m.def("run_knndbscan", &knndbscan_py, R"doc(
Perform kNN-DBSCAN clustering on a dataset using a precomputed k-nearest neighbors graph.

This function implements a variant of DBSCAN that uses a k-nearest neighbors (kNN) graph
for efficient density-based clustering. It supports parallel execution using OpenMP and
can be used in distributed environments with MPI.

Args:
    N (int): Total number of data points in the dataset.
    eps (float): Maximum distance (epsilon) for considering two points as neighbors.
                 Points within this distance are considered density-reachable.
    minPts (int): Minimum number of points required to form a dense region (core point).
    k (int): Number of nearest neighbors stored for each point (excluding self).
             Should be at least minPts - 1.
    JA (numpy.ndarray of int): Array of neighbor indices from the kNN graph.
                               Shape should be (N * k,), stored in row-major order.
    A (numpy.ndarray of float): Array of distances corresponding to the neighbors in JA.
                                Shape should be (N * k,), stored in row-major order.
    threads (int, optional): Number of OpenMP threads to use for parallel computation.
                             Defaults to 1 (single-threaded).
    verbose (bool, optional): Whether to print timing and debug information.
                              Defaults to False.

Returns:
    numpy.ndarray of int: Cluster labels for each point. Shape is (N,).
                          - Core points in clusters are assigned positive cluster IDs.
                          - Border points are assigned the same ID as their core point.
                          - Noise points are assigned -1.

Raises:
    RuntimeError: If MPI initialization fails or if input arrays have invalid shapes/sizes.

Note:
    The kNN graph should be precomputed and provided as JA and A arrays.
    The function assumes the graph is undirected and that distances are symmetric.
    MPI is initialized automatically if not already done, but not finalized to allow
    multiple calls within the same Python process.
)doc",
          py::arg("N"), py::arg("eps"), py::arg("minPts"), py::arg("k"), py::arg("JA"), py::arg("A"), py::arg("threads") = 1, py::arg("verbose") = false);
}
