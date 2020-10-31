from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt


def evaluate_clustering(clustering, classes):
    labels = clustering[:, -1]
    ars = metrics.adjusted_rand_score(classes, labels)
    fms = metrics.fowlkes_mallows_score(classes, labels)

    dbs = metrics.davies_bouldin_score(clustering[:, :-1], labels)
    ss = metrics.silhouette_score(clustering[:, :-1], labels)
    chs = metrics.calinski_harabasz_score(clustering[:, :-1], labels)

    print('External metrics')
    print('Adjusted rand score: ', ars)
    print('Fowlkes-Mallows score: ', fms)
    print('Internal metrics')
    print('Davies-Bouldin score: ', dbs)
    print('Silhouette score: ', ss)
    print('Calinski-Harabasz score: ', chs)


def evaluate_external(data, classes, algorithm):
    k_values = [2,3]
    ars_values = np.zeros([9])
    fms_values = np.zeros([9])
    for i in range(len(k_values)):
        k = k_values[i]
        print(f"Computing for k={k}")
        clustering = algorithm(data, k)
        ars_values[i] = metrics.adjusted_rand_score(classes, clustering[:, -1])
        fms_values[i] = metrics.fowlkes_mallows_score(classes, clustering[:, -1])
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title('Adjusted rand score')
    axes[0].bar(k_values, ars_values)
    axes[1].set_title('Fowlkes-Mallows score')
    axes[1].bar(k_values, fms_values)
    plt.show()


def evaluate_internal(data, algorithm):
    k_values = [2, 3, 4, 5, 6, 8, 10, 15, 20]
    sil_values = np.zeros([9])
    dbs_values = np.zeros([9])
    chs_values = np.zeros([9])
    for i in range(len(k_values)):
        k = k_values[i]
        print(f"Computing for k={k}")
        clustering = algorithm(data, k)
        sil_values[i] = metrics.silhouette_score(data, clustering[:, -1])
        dbs_values[i] = metrics.davies_bouldin_score(data, clustering[:, -1])
        chs_values[i] = metrics.calinski_harabasz_score(data, clustering[:, -1])
    fig, axes = plt.subplots(1, 3)
    axes[0].set_title('Davies-Bouldin score')
    axes[0].bar(k_values, dbs_values)
    axes[1].set_title('Calinski-Harabasz score')
    axes[1].bar(k_values, chs_values)
    axes[2].set_title('Silhouette score')
    axes[2].bar(k_values, sil_values)
    plt.show()

def evaluate_DBSCAN(db_labels, data, classes):
    labels = np.array(db_labels)
    print(np.max(labels,0))
    ars = metrics.adjusted_rand_score(classes, labels)
    fms = metrics.fowlkes_mallows_score(classes, labels)

    print('External metrics')
    print('Adjusted rand score: ', ars)
    print('Fowlkes-Mallows score: ', fms)
