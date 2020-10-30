from KHM import KHM
from k_means import kmeans
import read_datasets
import numpy as np
from evaluate import evaluate_clustering, evaluate_DBSCAN
from sklearn import metrics
from matplotlib import pyplot as plt

data, classes = read_datasets.read_waveform()
#data, classes = read_datasets.read_cn4()
#data, classes = read_datasets.read_adult()

k_values = [2,3,4,5,6,8,10,15,20]
ars_values = np.zeros([9])
fms_values = np.zeros([9])
dbs_values = np.zeros([9])
chs_values = np.zeros([9])
for i in range(len(k_values)):
    k = k_values[i]
    clustering = kmeans(data, k)
    ars_values[i] = metrics.adjusted_rand_score(classes, clustering[:, -1])
    fms_values[i] = metrics.fowlkes_mallows_score(classes, clustering[:, -1])
    dbs_values[i] = metrics.davies_bouldin_score(data, clustering[:, -1])
    chs_values[i] = metrics.calinski_harabasz_score(data, clustering[:, -1])
fig, axes = plt.subplots(2, 2)
axes[0,0].set_title('Adjusted rand score')
axes[0,0].bar(k_values, ars_values)
axes[0,1].set_title('Fowlkes-Mallows score')
axes[0,1].bar(k_values, fms_values)
axes[1,0].set_title('Davies-Bouldin score')
axes[1,0].bar(k_values, dbs_values)
axes[1,1].set_title('Calinski-Harabasz score')
axes[1,1].bar(k_values, chs_values)
plt.show()
