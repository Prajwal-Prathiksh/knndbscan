#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <mpi.h>

#include "../include/globals.h"
#include "clusters.cpp"

namespace py = pybind11;

py::array_t<int> knndbscan_serial(int N, float eps, int minPts, int k, py::array_t<int> JA_np, py::array_t<float> A_np) {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    
    auto JA_buf = JA_np.request();
    auto A_buf = A_np.request();
    point_int* JA = static_cast<point_int*>(JA_buf.ptr);
    float* A = static_cast<float*>(A_buf.ptr);
    
    std::vector<point_int> labels = knndbscan(N, eps, minPts, k, JA, A);
    
    MPI_Finalize();
    
    return py::array_t<int>(labels.size(), labels.data());
}

PYBIND11_MODULE(_core, m) {
    m.def("knndbscan", &knndbscan_serial, "kNN-DBSCAN clustering",
          py::arg("N"), py::arg("eps"), py::arg("minPts"), py::arg("k"), py::arg("JA"), py::arg("A"));
}
