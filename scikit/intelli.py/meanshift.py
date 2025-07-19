import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import estimate_bandwidth

coordinates = [[2,2,3],[6,7,8],[5,14,16]]
x, _=make_blobs(n_samples=150, centers=coordinates, cluster_std=0.60)
data_fig = plt.figure(figsize=(8,8))
a = data_fig.add_subplot(111,projection='3d')
a.scatter(x[:,0],x[:,1],x[:,2],marker='o',color='purple')
# plt.show()
bw = estimate_bandwidth(x,quantile=0.2, n_samples=500)
# print(bw)
msc = MeanShift(bandwidth=bw, bin_seeding=True)
msc.fit(x)
cluster_centers = msc.cluster_centers_
labels = msc.labels_
cluster_labels = np.unique(labels)
n_clusters = len(cluster_labels)
# print(n_clusters)

fig = plt.figure(figsize=(8,8))
b = data_fig.add_subplot(111,projection='3d')
b.scatter(cluster_centers[:,0], cluster_centers[:,1],cluster_centers[:,2], marker='o', color='green',s=300, linewidths=5, zorder=10)
plt.title("estimated np. of clusters:%d" %n_clusters)
plt.show()



