from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def compareDBSCAN_alg(data, eps):
    data = StandardScaler().fit_transform(data)
    auto_clustering = DBSCAN(eps=eps, algorithm='auto').fit(data)
    print('Auto:')
    print(auto_clustering.labels_)
    plot_result(auto_clustering, data)
    ball_tree_clustering = DBSCAN(algorithm='ball_tree', eps=eps).fit(data)
    print('Ball tree:')
    print(ball_tree_clustering.labels_)
    plot_result(ball_tree_clustering, data)
    kd_tree_clustering = DBSCAN(algorithm='kd_tree', eps=eps).fit(data)
    print('KD tree:')
    print(kd_tree_clustering.labels_)
    plot_result(kd_tree_clustering, data)
    try:
        brute_clustering = DBSCAN(algorithm='brute', eps=eps, metric='cosine').fit(data)
        print('Brute:')
        print(brute_clustering.labels_)
        plot_result(brute_clustering, data)
    except:
        print("Brute clustering could not be performed (too may data)")

def compareDBSCAN_metric(data, eps):
    data = StandardScaler().fit_transform(data)
    euclidean_clustering = DBSCAN(metric='euclidean', eps=eps).fit(data)
    print('Euclidean:')
    print(euclidean_clustering.labels_)
    plot_result(euclidean_clustering, data)
    cosine_clustering = DBSCAN(metric='cosine', eps=eps).fit(data)
    print('Cosine:')
    print(cosine_clustering.labels_)
    plot_result(cosine_clustering, data)
    l1_clustering = DBSCAN(metric='l1', eps=eps).fit(data)
    print('L1:')
    print(l1_clustering.labels_)
    plot_result(cosine_clustering, data)
    l2_clustering = DBSCAN(metric='l2', eps=eps).fit(data)
    print('L2:')
    print(l2_clustering.labels_)
    plot_result(cosine_clustering, data)
    man_clustering = DBSCAN(metric='manhattan', eps=eps).fit(data)
    print('Manhattan:')
    print(man_clustering.labels_)
    plot_result(cosine_clustering, data)


def plot_result(db, data):
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = [plt.get_cmap('Spectral')(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
