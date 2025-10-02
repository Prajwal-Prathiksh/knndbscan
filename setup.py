import subprocess

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def get_mpi_flags():
    """Get MPI flags using mpicxx compiler wrapper."""
    try:
        compile_flags = subprocess.check_output(
            ["mpicxx", "--showme:compile"], text=True
        ).split()
        link_flags = subprocess.check_output(
            ["mpicxx", "--showme:link"], text=True
        ).split()

        # Keep only flags (starting with -)
        return (
            [f for f in compile_flags if f.startswith("-")],
            [f for f in link_flags if f.startswith("-")],
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Using fallback MPI configuration")
        return ["-I/usr/include/mpi"], ["-lmpi"]


mpi_compile_flags, mpi_link_flags = get_mpi_flags()

ext_modules = [
    Pybind11Extension(
        "knndbscan._core",
        ["src/pybind.cpp", "src/clusters.cpp"],
        include_dirs=["include"],
        extra_compile_args=[
            "-O3",
            "-fopenmp",
            "-Wno-unused-variable",
            "-Wno-unused-value",
            "-Wno-sign-compare",
        ]
        + mpi_compile_flags,
        extra_link_args=["-fopenmp"] + mpi_link_flags,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
