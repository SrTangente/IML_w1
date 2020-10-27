from sklearn import metrics


def evaluate_clustering(clustering, classes):
    labels = clustering[:, -1]
    ars = metrics.adjusted_rand_score(classes, labels)
    fms = metrics.fowlkes_mallows_score(classes, labels)

    dbs = metrics.davies_bouldin_score(clustering, labels)
    ss = metrics.silhouette_score(clustering, labels)

    print('External metrics')
    print('Adjusted rand score: ', ars)
    print('Fowlkes-Mallows score: ', fms)
    print('Internal metrics')
    print('Davies-Bouldin score: ', dbs)
    print('Silhouette score: ', ss)

def evaluate_DBSCAN(db_labels,data, classes):
    ars = metrics.adjusted_rand_score(classes, db_labels)
    fms = metrics.fowlkes_mallows_score(classes, db_labels)

    print('External metrics')
    print('Adjusted rand score: ', ars)
    print('Fowlkes-Mallows score: ', fms)
