#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <mpi.h>
#include <omp.h>

#include "../include/globals.h"
#include "clusters.cpp"

namespace py = pybind11;

py::array_t<int> knndbscan_py(int N, float eps, int minPts, int k, py::array_t<int> JA_np, py::array_t<float> A_np, int threads = 1) {
    int initialized;
    MPI_Initialized(&initialized);
    
    int provided;
    if (!initialized) {
        MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    }
    
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(threads);
    
    auto JA_buf = JA_np.request();
    auto A_buf = A_np.request();
    point_int* JA = static_cast<point_int*>(JA_buf.ptr);
    float* A = static_cast<float*>(A_buf.ptr);
    
    std::vector<point_int> labels = knndbscan(N, eps, minPts, k, JA, A);
    
    // Don't finalize MPI here - let it persist for multiple calls
    // MPI will be finalized when the Python process exits
    
    return py::array_t<int>(labels.size(), labels.data());
}

PYBIND11_MODULE(_core, m) {
    m.def("knndbscan", &knndbscan_py, "kNN-DBSCAN clustering",
          py::arg("N"), py::arg("eps"), py::arg("minPts"), py::arg("k"), py::arg("JA"), py::arg("A"), py::arg("threads") = 1);
}
