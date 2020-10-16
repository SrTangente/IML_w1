from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def compareDBSCAN_alg(data):
    ball_tree_clustering = DBSCAN(algorithm='ball_tree').fit(data)
    print('Ball tree:')
    plot_result(ball_tree_clustering, data)
    print(ball_tree_clustering.labels_)
    kd_tree_clustering = DBSCAN(algorithm='kd_tree').fit(data)
    print('KD tree:')
    print(kd_tree_clustering.labels_)
    try:
        brute_clustering = DBSCAN(algorithm='brute').fit(data)
        print('Brute:')
        print(brute_clustering.labels_)
    except:
        print("Brute clustering could not be performed (too may data)")

def compareDBSCAN_metric(data):
    euclidean_clustering = DBSCAN(metric='euclidean').fit(data)
    print('Euclidean:')
    print(euclidean_clustering.labels_)
    cosine_clustering = DBSCAN(metric='cosine').fit(data)
    print('Cosine:')
    print(cosine_clustering.labels_)
    l1_clustering = DBSCAN(metric='l1').fit(data)
    print('L1:')
    print(l1_clustering.labels_)
    l2_clustering = DBSCAN(metric='l2').fit(data)
    print('L2:')
    print(l2_clustering.labels_)
    man_clustering = DBSCAN(metric='Manhattan').fit(data)
    print('Manhattan:')
    print(man_clustering.labels_)


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
