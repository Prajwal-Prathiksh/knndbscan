"""knndbscan: Parallel kNN-DBSCAN clustering implementation."""

import os

# Configure OpenMPI environment for optimal single-node performance
# These settings prevent common errors on HPC clusters (e.g., TACC) where
# the 'smcuda' (Shared Memory CUDA) BTL component can fail to initialize.
# Users can override these by setting KNNDBSCAN_AUTO_CONFIGURE_MPI=0 before importing.
_AUTO_CONFIGURE_MPI = os.environ.get("KNNDBSCAN_AUTO_CONFIGURE_MPI", "1") == "1"

if _AUTO_CONFIGURE_MPI:
    # Handle OMPI_MCA_btl - append smcuda to exclusion list if not already excluded
    btl_setting = os.environ.get("OMPI_MCA_btl", "")
    if "smcuda" not in btl_setting:
        if btl_setting.startswith("^"):
            # Already has exclusions, append to them
            components_to_exclude = btl_setting[1:]  # Remove leading ^
            if components_to_exclude:
                os.environ["OMPI_MCA_btl"] = f"^{components_to_exclude},smcuda"
            else:
                os.environ["OMPI_MCA_btl"] = "^smcuda"
        else:
            # No exclusions yet, or has inclusions - add smcuda exclusion
            os.environ["OMPI_MCA_btl"] = "^openib,smcuda"


# Import C++ extension after configuring environment
from ._core import run_knndbscan
from .runner import knn_dbscan

__version__ = "0.1.0"
__all__ = ["knn_dbscan", "run_knndbscan"]
