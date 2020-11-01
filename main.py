from read_datasets import read_cn4, read_adult, read_waveform
from k_means import kmeans
from bisecting_k_means import bisecting_kmeans
from KHM import KHM
from FCM import FCM
from evaluate import evaluate_clustering, evaluate_DBSCAN
from CompareDBSCAN import compareDBSCAN_alg
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler


def input_dataset():
    print("Select dataset to which execute the unsupervised algorithms:")
    print("1 adult")
    print("2 waveform")
    print("3 cn4")
    num_dataset = 0
    while True:
        try:
            num_dataset = int(input("Dataset number: "))
        except:
            print("Incorrect value. It has to be one of (1, 2, 3).")
            continue
        if num_dataset in range(1, 4):
            break
        else:
            print("Incorrect value. It has to be one of (1, 2, 3).")

    return num_dataset


def input_algorithm():
    print("------------------------")
    print("Select algorithm to use:")
    print("1 DBSCAN")
    print("2 K-Means")
    print("3 Bisecting K-Means")
    print("4 K-Harmonic Means")
    print("5 Fuzzy C-Means")
    num_algorithm = 0
    while True:
        try:
            num_algorithm = int(input("Algorithm number: "))
        except:
            print("Incorrect value. It has to be one of (1, 2, 3, 4, 5).")
            continue
        if num_algorithm in range(1, 6):
            break
        else:
            print("Incorrect value. It has to be one of (1, 2, 3, 4, 5).")

    return num_algorithm


def read_dataset(num_dataset):
    if num_dataset == 1:
        return read_adult(), 2
    elif num_dataset == 2:
        return read_waveform(), 3
    else:
        return read_cn4(), 3


def execute_algorithm(num_dataset, num_algorithm):
    """
    Execute the selected algorithm on the selected dataset with the best parameters obtained
    :param num_dataset: the number of the dataset selected
    :param num_algorithm: the number of the dataset selected
    :return: the tagged_data predicted and the real classes of the dataset
    """
    (data, classes), k = read_dataset(num_dataset)
    print("Executing algorithm...")
    print("This process can take some minutes")
    if num_algorithm == 1:
        data = StandardScaler().fit_transform(data)
        if num_dataset == 1:
            tags = (DBSCAN(eps=0.09, min_samples=np.log(len(data))).fit(data)).labels_
        elif num_dataset == 2:
            tags = (DBSCAN(eps=0.29, min_samples=np.log(len(data)), metric='cosine').fit(data)).labels_
        else:
            tags = (DBSCAN(eps=0.005, min_samples=np.log(len(data))).fit(data)).labels_
        tagged_data = np.zeros((data.shape[0], data.shape[1] + 1))
        tagged_data[:, :-1] = data
        tagged_data[:, -1] = tags
    elif num_algorithm == 2:
        tagged_data = kmeans(data, k)
    elif num_algorithm == 3:
        tagged_data = bisecting_kmeans(data, k)
    elif num_algorithm == 4:
        if num_dataset == 1:
            tagged_data = KHM(data, k, 3, 100, 10e-12)
        elif num_dataset == 2:
            tagged_data = KHM(data, k, 2, 100, 10e-12)
        else:
            tagged_data = KHM(data, k, 3, 100, 10e-12)
    else:
        if num_dataset == 1:
            tagged_data = FCM(data, k, 3, 100, 10e-12)
        elif num_dataset == 2:
            tagged_data = FCM(data, k, 2, 100, 10e-12)
        else:
            tagged_data = FCM(data, k, 3, 100, 10e-12)
    return tagged_data, classes


if __name__ == "__main__":

    num_dataset = input_dataset()
    num_algorithm = input_algorithm()
    tagged_data, classes = execute_algorithm(num_dataset, num_algorithm)
    evaluate_clustering(tagged_data, classes)
