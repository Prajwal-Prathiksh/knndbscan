import sys
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

from knndbscan import _core

# Get threads from command line
threads = int(sys.argv[1]) if len(sys.argv) > 1 else 16

# Constants
N_SAMPLES = 100_000
NOISE = 0.05
RANDOM_STATE = 42
EPS = 100
MIN_PTS = 20

print(f"Starting k-NN DBSCAN test with {N_SAMPLES} samples, {threads} threads")

# Start total timing
total_start = time.perf_counter()

# Generate custom 32D dataset with multiple clusters
data_gen_start = time.perf_counter()
np.random.seed(RANDOM_STATE)
n_features = 100
n_clusters = 20
cluster_size = N_SAMPLES // n_clusters

X = []
y_true = []

for i in range(n_clusters):
    # Create cluster centers far apart in 32D space
    center = np.random.randn(n_features) * 10  # Spread centers
    center[i % n_features] += 50  # Make them more separated

    # Generate points around each center
    cluster_points = np.random.multivariate_normal(
        center, np.eye(n_features) * (1 + i), cluster_size  # Different variances
    )
    X.append(cluster_points)
    y_true.extend([i] * cluster_size)

X = np.vstack(X)
# Add noise
X += np.random.randn(*X.shape) * NOISE

data_gen_end = time.perf_counter()
print(f"Data generation: {data_gen_end - data_gen_start:.3f} seconds")
print(
    f"Generated {X.shape[0]} samples with {X.shape[1]} features in {n_clusters} clusters."
)

# Build k-NN graph
knn_start = time.perf_counter()
nn = NearestNeighbors(n_neighbors=MIN_PTS, algorithm="auto")
nn.fit(X)
distances, indices = nn.kneighbors(X)
knn_end = time.perf_counter()
print(f"k-NN graph building: {knn_end - knn_start:.3f} seconds")

# Prepare data
prep_start = time.perf_counter()
A = distances[:, 1:].flatten().astype(np.float32)
JA = indices[:, 1:].flatten().astype(np.int32)
k = MIN_PTS - 1
prep_end = time.perf_counter()
print(f"Data preparation: {prep_end - prep_start:.3f} seconds")
print(f"Memory usage: {A.nbytes + JA.nbytes} bytes")

# Run clustering
cluster_start = time.perf_counter()
labels_pred = _core.knndbscan(N_SAMPLES, EPS, MIN_PTS, k, JA, A, threads=threads)
cluster_end = time.perf_counter()
print(f"Clustering: {cluster_end - cluster_start:.2f} seconds")
print(f"Number of clusters: {len(set(labels_pred)) - (1 if -1 in labels_pred else 0)}")

# Total timing
total_end = time.perf_counter()
total_time = total_end - total_start

print(f"\nTotal execution time: {total_time:.3f} seconds")
print("Time breakdown (%):")
print(f"  Data generation: {(data_gen_end - data_gen_start)/total_time*100:.1f}%")
print(f"  k-NN graph: {(knn_end - knn_start)/total_time*100:.1f}%")
print(f"  Data prep: {(prep_end - prep_start)/total_time*100:.1f}%")
print(f"  Clustering: {(cluster_end - cluster_start)/total_time*100:.1f}%")
