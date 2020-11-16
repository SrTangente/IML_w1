import time

from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt
from sklearn import decomposition
from pca import pca

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
    k_values = [5, 7, 9, 11, 13, 15]
    ars_values = np.zeros([len(k_values)])
    fms_values = np.zeros([len(k_values)])
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
    k_values = [5, 7, 9, 11, 13, 15]
    sil_values = np.zeros([len(k_values)])
    dbs_values = np.copy(sil_values)
    chs_values = np.copy(sil_values)
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


def evaluate_scatter(data, labels, algorithm, k_values, p_values):
    ars_values = np.zeros([len(k_values), len(p_values)])
    fms_values = np.copy(ars_values)
    sil_values = np.copy(ars_values)
    dbs_values = np.copy(ars_values)
    chs_values = np.copy(ars_values)
    original_data = np.copy(data)
    transformed_data = pca(data)
    for i in range(len(p_values)):
        p = p_values[i]
        if p == 0:
            data = original_data
        else:
            data = transformed_data[:, :p]
        for j in range(len(k_values)):
            start = time.time()
            k = k_values[j]
            print(f"Computing for {p} components and k={k}")
            clustering = algorithm(data, k)
            alg_time = time.time()
            print(f'Clustering time:{alg_time - start}')
            ars_values[j, i] = metrics.adjusted_rand_score(labels, clustering[:, -1])
            fms_values[j, i] = metrics.fowlkes_mallows_score(labels, clustering[:, -1])
            sil_values[j, i] = metrics.silhouette_score(data, clustering[:, -1])
            dbs_values[j, i] = metrics.davies_bouldin_score(data, clustering[:, -1])
            chs_values[j, i] = metrics.calinski_harabasz_score(data, clustering[:, -1])
            print(f'Evaluation time:{time.time() - alg_time}')
    print('Adjusted rand index values: ')
    print(ars_values)
    print('Fowlkes-Mallows values: ')
    print(fms_values)
    print('Silhouette values: ')
    print(sil_values)
    print('Davies-Bouldin values: ')
    print(dbs_values)
    print('Calinski-Harabasz values: ')
    print(chs_values)
    fig1, axes1 = plt.subplots(1, 2, figsize=(6.66, 3.33))
    axes1[0].set_title('Adjusted rand score')
    for j in range(len(p_values)):
        axes1[0].scatter(k_values, ars_values[:, j])
    axes1[0].set_xlabel('k')
    axes1[0].legend([f'p={p}' for p in p_values])
    axes1[1].set_title('Fowlkes-Mallows score')
    for j in range(len(p_values)):
        axes1[1].scatter(k_values, fms_values[:, j])
    axes1[1].set_xlabel('k')
    axes1[0].legend([f'p={p}' for p in p_values])
    plt.show()

    fig2, axes2 = plt.subplots(1, 3, figsize=(10, 3.33))
    axes2[0].set_title('Davies-Bouldin score')
    for j in range(len(p_values)):
        axes2[0].scatter(k_values, dbs_values[:, j])
    axes1[0].legend([f'p={p}' for p in p_values])

    axes2[1].set_title('Calinski-Harabasz score')

    for j in range(len(p_values)):
        axes2[1].scatter(k_values, chs_values[:, j])
    axes1[0].legend([f'p={p}' for p in p_values])
    axes2[1].set_xlabel('k')

    axes2[2].set_title('Silhouette score')
    for j in range(len(p_values)):
        axes2[2].scatter(k_values, sil_values[:, j])
    axes1[0].legend([f'p={p}' for p in p_values])
    axes2[2].set_xlabel('k')
    plt.show()


def evaluate_DBSCAN(db_labels, data, classes):
    labels = np.array(db_labels)
    print("Number of labels: ", np.max(labels, 0))
    if np.max(labels) != -1:
        ars = metrics.adjusted_rand_score(classes, labels)
        fms = metrics.fowlkes_mallows_score(classes, labels)
        sil = metrics.silhouette_score(data, labels)
        dbs = metrics.davies_bouldin_score(data, labels)
        chs = metrics.calinski_harabasz_score(data, labels)

        print('Internal metrics')
        print('Silhouette score: ', sil)
        print('Davies-Bouldin score: ', dbs)
        print('Calinski-Harabasz score: ', chs)
        print('---------------')
        print('External metrics')
        print('Adjusted rand score: ', ars)
        print('Fowlkes-Mallows score: ', fms)
    else:
        print('No se ha encontrado ningún cluster')
