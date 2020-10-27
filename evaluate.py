from sklearn import metrics
import numpy as np


def evaluate_clustering(clustering, classes):
    labels = clustering[:, -1]
    ars = metrics.adjusted_rand_score(classes, labels)
    fms = metrics.fowlkes_mallows_score(classes, labels)

    dbs = metrics.davies_bouldin_score(clustering[:, :-1], labels)
    ss = metrics.silhouette_score(clustering[:, :-1], labels)

    print('External metrics')
    print('Adjusted rand score: ', ars)
    print('Fowlkes-Mallows score: ', fms)
    print('Internal metrics')
    print('Davies-Bouldin score: ', dbs)
    print('Silhouette score: ', ss)

def evaluate_DBSCAN(db_labels, data, classes):
    print(classes)
    labels = np.array(db_labels)
    print(labels)
    ars = metrics.adjusted_rand_score(classes, labels)
    fms = metrics.fowlkes_mallows_score(classes, labels)

    print('External metrics')
    print('Adjusted rand score: ', ars)
    print('Fowlkes-Mallows score: ', fms)
