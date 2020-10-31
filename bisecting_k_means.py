import numpy as np
from k_means import kmeans
from read_datasets import read_waveform, read_adult, read_cn4
from evaluate import evaluate_clustering
from sklearn.metrics import euclidean_distances


def bisecting_kmeans(data, k):
    """
    Perform a bisecting k-means on the given data
    :param data: numpy.ndarray representing the untagged data to cluster
    :param k: number of clusters
    :return: the same data with the corresponding class associated
    """
    # Initialise all elements to the same cluster (cluster 1)
    n = np.size(data, 0)
    n_prop = np.size(data, 1)
    tagged_data = np.zeros([n, n_prop + 1])
    tagged_data[:, :-1] = data

    centroids = [0]

    # In the first iteration, split the only existing cluster
    cluster_to_split = 0.0
    # Divide one cluster in two. Repeat it 'k-1' times to get k clusters
    for i in range(k - 1):

        # Select the data to split (without the class associated)
        data_to_split = tagged_data[tagged_data[:, -1] == cluster_to_split][:, :-1]
        # Split the cluster using k-means (bisecting step)
        new_tagged_data, new_centroids = kmeans(data_to_split, 2, return_centroids=True)
        centroids[int(cluster_to_split)] = new_centroids[0]
        centroids.append(new_centroids[1])
        new_tags = new_tagged_data[:, -1]
        # The returning values of kmeans with class '0' will keep its original cluster number
        # and the ones with class '1' will get the tag corresponding to the current iteration
        new_tags = np.array([cluster_to_split if tag == 0. else float(i + 1) for tag in new_tags])
        # Update the tags from the tagged_data
        sel = tagged_data[:, -1] == cluster_to_split
        rows_affected = tagged_data[sel]
        rows_affected[:, -1] = new_tags
        tagged_data[sel] = rows_affected
        print(tagged_data[:, -1])
        print(np.bincount(tagged_data[:, -1].astype(int)))

        # Select the cluster to split for the next iteration (the most heterogeneous)
        max_distance = 0.0
        cluster_to_split = 0.0
        for num_cluster in range(i + 1):
            cluster = tagged_data[tagged_data[:, -1] == float(num_cluster)][:, :-1]
            mean_distance = float(sum(euclidean_distances(cluster, centroids[num_cluster].reshape(1, -1)))) / float(
                cluster.shape[0])
            if mean_distance > max_distance:
                cluster_to_split = float(num_cluster)
                max_distance = mean_distance

    return tagged_data
