from KHM import KHM
from k_means import kmeans
import read_datasets
import numpy as np
from evaluate import evaluate_clustering, evaluate_DBSCAN
from CompareDBSCAN import compareDBSCAN_alg, compareDBSCAN_metric
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

waveform_data, waveform_classes = read_datasets.read_adult()

compareDBSCAN_metric(waveform_data, waveform_classes, 0.5, np.log(np.size(waveform_data, 0)))

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(waveform_data)
distances, indices = nbrs.kneighbors(waveform_data)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.plot(distances)
plt.show()

#evaluate_clustering(kmeans(waveform_data, 2), waveform_classes)


