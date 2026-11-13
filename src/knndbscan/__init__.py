"""kNN-DBSCAN: density-based clustering of kNN graphs, with a parallel C++ core.

Implements the algorithm of Chen, Ruys and Biros, "KNN-DBSCAN: a DBSCAN in high
dimensions" (arXiv:2009.04552).
"""

from importlib.metadata import version

from .cluster import KNNDBSCAN, knn_dbscan, knn_dbscan_from_graph

__version__ = version("knndbscan")
__all__ = ["KNNDBSCAN", "knn_dbscan", "knn_dbscan_from_graph"]
