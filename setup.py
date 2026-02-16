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
        # MPI compiler not found. Using fallback MPI configuration
        return ["-I/usr/include/mpi"], ["-lmpi"]


def get_openmp_flag():
    """Detect OpenMP flag based on compiler."""
    try:
        # Check if using IBM XL compiler
        version_output = subprocess.check_output(
            ["mpicxx", "--version"], text=True, stderr=subprocess.STDOUT
        )
        if "xl" in version_output.lower() or "ibm" in version_output.lower():
            return "-qopenmp"
    except Exception:
        pass

    # Default to GCC/Clang flag
    return "-fopenmp"


mpi_compile_flags, mpi_link_flags = get_mpi_flags()
openmp_flag = get_openmp_flag()

ext_modules = [
    Pybind11Extension(
        "knndbscan._core",
        ["src/pybind.cpp", "src/clusters.cpp"],
        include_dirs=["include"],
        extra_compile_args=[
            "-O3",
            openmp_flag,
            "-Wno-unused-variable",
            "-Wno-unused-value",
            "-Wno-sign-compare",
        ]
        + mpi_compile_flags,
        extra_link_args=[openmp_flag] + mpi_link_flags,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
