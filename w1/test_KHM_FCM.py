from KHM import KHM
from FCM import FCM
import numpy as np
from read_datasets import *
from evaluate import *

algorithm=FCM
data,labels=read_adult()


k_values = [2, 3, 4, 5,6]
p_values = [2, 3, 4]
ars_values = np.zeros([5, 3])
fms_values = np.zeros([5, 3])
sil_values = np.zeros([5, 3])
dbs_values = np.zeros([5, 3])
chs_values = np.zeros([5, 3])
for i in range(len(k_values)):
    for j in range(len(p_values)):
        start=time.time()
        k = k_values[i]
        p = p_values[j]
        print(f"Computing for k={k} p={p}")
        clustering = algorithm(data, k, p)
        alg_time=time.time()
        print(f'Clustering time:{alg_time-start}')
        ars_values[i, j] = metrics.adjusted_rand_score(labels, clustering[:, -1])
        fms_values[i, j] = metrics.fowlkes_mallows_score(labels, clustering[:, -1])
        sil_values[i,j] = metrics.silhouette_score(data, clustering[:, -1])
        dbs_values[i,j] = metrics.davies_bouldin_score(data, clustering[:, -1])
        chs_values[i,j] = metrics.calinski_harabasz_score(data, clustering[:, -1])
        print(f'Evaluation time:{time.time() - alg_time}')
fig1, axes1 = plt.subplots(1, 2, figsize=(6.66, 3.33))
axes1[0].set_title('Adjusted rand score')
for j in range(len(p_values)):
    axes1[0].scatter(k_values, ars_values[:,j])
axes1[0].set_xlabel('k')
axes1[0].legend(['p=2','p=3','p=4'])

axes1[1].set_title('Fowlkes-Mallows score')
for j in range(len(p_values)):
    axes1[1].scatter(k_values, fms_values[:,j])
axes1[1].set_xlabel('k')
axes1[1].legend(['p=2','p=3','p=4'])
plt.show()

fig2, axes2 = plt.subplots(1, 3, figsize=(10, 3.33))
axes2[0].set_title('Davies-Bouldin score')
for j in range(len(p_values)):
    axes2[0].scatter(k_values, dbs_values[:,j])
axes2[0].legend(['p=2','p=3','p=4'])


axes2[1].set_title('Calinski-Harabasz score')

for j in range(len(p_values)):
    axes2[1].scatter(k_values, chs_values[:,j])
axes2[1].legend(['p=2','p=3','p=4'])
axes2[1].set_xlabel('k')

axes2[2].set_title('Silhouette score')
for j in range(len(p_values)):
    axes2[2].scatter(k_values, sil_values[:,j])
axes2[2].legend(['p=2','p=3','p=4'])
axes2[2].set_xlabel('k')

plt.show()