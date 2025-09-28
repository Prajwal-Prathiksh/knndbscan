from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "knndbscan._core",
        ["src/pybind.cpp", "src/clusters.cpp"],
        include_dirs=["include"],
        libraries=["mpi", "gomp"],  # MPI and OpenMP libraries
        extra_compile_args=[
            "-O3",
            "-fopenmp",
            "-Wno-unused-variable",
            "-Wno-unused-value",
            "-Wno-sign-compare",
        ],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="knndbscan",
    version="0.1.0",
    packages=["knndbscan"],
    package_dir={"": "."},
    ext_modules=ext_modules,
)
