import numpy as np
from sklearn import metrics

file1 = file_name_ground_truth
labels_true = np.loadtxt(file1)
file2 = file_name_to_check
labels = np.loadtxt(file2)

n = 70000 # number of points
NMI = metrics.normalized_mutual_info_score(labels, labels_true, average_method='arithmetic')

clustered = (labels >= 0)
ratio_points_clustered = np.sum(clustered) / n
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print("number of clusters: %d" % n_clusters)
print("number of noise: %d" % n_noise)
print("ratio of points clustered: %0.3f" % ratio_points_clustered)
print("NMI value: %0.3f" % NMI)



