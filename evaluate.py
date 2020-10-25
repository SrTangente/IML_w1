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
    print('Davies-Bouldin score: ', dbs)
    print('Internal metrics')
    print('Silhouette score: ', ss)
