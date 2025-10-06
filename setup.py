import os
import subprocess

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def check_env_flag(name: str) -> bool:
    """Check if environment variable is set to enable a feature."""
    return os.getenv(name, "0").upper() in ["1", "ON", "YES", "TRUE"]


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


mpi_compile_flags, mpi_link_flags = get_mpi_flags()

# Build configuration based on environment variables
extra_compile_args = [
    "-fopenmp",
    "-Wno-unused-variable",
    "-Wno-unused-value",
    "-Wno-sign-compare",
    "-O3",
]

# Build with debug symbols
if check_env_flag("DEBUG"):
    extra_compile_args.append("-g")


ext_modules = [
    Pybind11Extension(
        "knndbscan._core",
        ["src/pybind.cpp", "src/clusters.cpp"],
        include_dirs=["include"],
        extra_compile_args=extra_compile_args + mpi_compile_flags,
        extra_link_args=["-fopenmp"] + mpi_link_flags,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
