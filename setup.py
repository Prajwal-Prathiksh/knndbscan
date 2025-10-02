from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "knndbscan._core",
        [
            "src/pybind.cpp",
            "src/clusters.cpp",
        ],
        include_dirs=["include"],
        libraries=["mpi", "gomp"],
        extra_compile_args=[
            "-O3",
            # "-g",  # Enable debugging symbols
            "-fopenmp",
            "-Wno-unused-variable",
            "-Wno-unused-value",
            "-Wno-sign-compare",
        ],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
