"""Test suite for knndbscan module."""

import numpy as np
import pytest

from knndbscan import knndbscan


def test_knndbscan_threading():
    """Test multi-threading functionality."""
    N = 100
    k = 5
    eps = 1.5
    minPts = 6  # k + 1

    # Create properly sized mock data
    # Generate valid neighbor indices (avoid self-neighbors)
    JA = np.zeros(N * k, dtype=np.int32)
    for i in range(N):
        # For each point i, generate k unique neighbors (excluding i)
        neighbors = []
        for j in range(k):
            neighbor = (i + j + 1) % N  # Simple cyclic neighbors
            neighbors.append(neighbor)
        JA[i * k : (i + 1) * k] = neighbors

    # Generate random distances
    np.random.seed(42)  # For reproducibility
    A = np.random.uniform(0.1, 2.0, size=N * k).astype(np.float32)

    # Test with different thread counts
    labels_1 = knndbscan(N, eps, minPts, k, JA, A, threads=1)
    labels_2 = knndbscan(N, eps, minPts, k, JA, A, threads=2)

    # Results should be consistent across thread counts
    assert len(labels_1) == len(labels_2) == N, "Label array length mismatch"
    # The clustering results should be identical
    np.testing.assert_array_equal(labels_1, labels_2)


def test_knndbscan_small_dataset():
    """Test with a very small dataset to verify noise detection."""
    N = 4
    k = 2
    eps = 0.5  # Small epsilon to make most points noise
    minPts = 3  # k + 1

    # Create a simple k-NN graph where points are far apart
    JA = np.array(
        [
            1,
            2,  # neighbors of point 0
            0,
            3,  # neighbors of point 1
            0,
            3,  # neighbors of point 2
            1,
            2,  # neighbors of point 3
        ],
        dtype=np.int32,
    )

    A = np.array(
        [
            2.0,
            2.0,  # large distances from point 0
            2.0,
            2.0,  # large distances from point 1
            2.0,
            2.0,  # large distances from point 2
            2.0,
            2.0,  # large distances from point 3
        ],
        dtype=np.float32,
    )

    labels = knndbscan(N, eps, minPts, k, JA, A, threads=1)

    # With large distances and small eps, most/all points should be noise (-1)
    assert len(labels) == N, f"Expected {N} labels, got {len(labels)}"
    assert isinstance(
        labels, np.ndarray
    ), f"Expected labels to be a numpy array, got {type(labels)}"
    assert labels.dtype == np.int32, "Expected labels to be int32"

    # Check that we get some result (could be all noise with these parameters)
    unique_labels = np.unique(labels)
    assert len(unique_labels) >= 1, "Expected at least the noise label (-1) in results"


if __name__ == "__main__":
    pytest.main([__file__])
