import os
import platform
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


def get_openmp_flags():
    """Detect OpenMP flags based on platform and compiler.

    Returns a tuple of (compile_flags, link_flags).
    """
    if platform.system() == "Darwin":
        # Apple clang does not support -fopenmp natively.
        # Requires libomp installed via Homebrew: brew install libomp
        try:
            brew_prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"], text=True
            ).strip()
            compile_flags = [
                f"-I{brew_prefix}/include",
                "-Xclang",
                "-fopenmp",
            ]
            link_flags = [f"-L{brew_prefix}/lib", "-lomp"]
            return compile_flags, link_flags
        except (subprocess.CalledProcessError, FileNotFoundError):
            # libomp not found via brew; fall back to best-effort flags
            return ["-Xclang", "-fopenmp"], ["-lomp"]

    try:
        # Check if using IBM XL compiler
        version_output = subprocess.check_output(
            ["mpicxx", "--version"], text=True, stderr=subprocess.STDOUT
        )
        if "xl" in version_output.lower() or "ibm" in version_output.lower():
            return ["-qopenmp"], ["-qopenmp"]
    except Exception:
        pass

    # Default to GCC/Clang flag
    return ["-fopenmp"], ["-fopenmp"]


mpi_compile_flags, mpi_link_flags = get_mpi_flags()
openmp_compile_flags, openmp_link_flags = get_openmp_flags()

# Build configuration based on environment variables
extra_compile_args = [
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
        extra_link_args=openmp_link_flags + mpi_link_flags,
        extra_compile_args=extra_compile_args
        + openmp_compile_flags
        + mpi_compile_flags,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
