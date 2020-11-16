from read_datasets import read_cn4, read_adult, read_waveform, read_vowel
from k_means import kmeans
from bisecting_k_means import bisecting_kmeans
from KHM import KHM
from FCM import FCM
from evaluate import evaluate_clustering, evaluate_DBSCAN, evaluate_scatter
from CompareDBSCAN import compareDBSCAN_alg
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.preprocessing import StandardScaler
from visualize_km_pca_tsne import visualize_km


def input_dataset():
    print("Select dataset to which execute the unsupervised algorithms:")
    print("1 adult")
    print("2 waveform")
    print("3 vowel")
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
    print("1 Compare PCA effects on k-means")
    print("2 Visualize PCA vs t-SNE")
    num_algorithm = 0
    while True:
        try:
            num_algorithm = int(input("Algorithm number: "))
        except:
            print("Incorrect value. It has to be one of (1, 2).")
            continue
        if num_algorithm in range(1, 2):
            break
        else:
            print("Incorrect value. It has to be one of (1, 2).")

    return num_algorithm


def read_dataset(num_dataset):
    if num_dataset == 1:
        return read_adult(), 2
    elif num_dataset == 2:
        return read_waveform(), 3
    else:
        return read_vowel(), 3


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
        if num_dataset < 3:
            k_values = [2, 3, 4]
            p_values = [0, 20, 10, 5]
        else:
            k_values = [7, 11, 15]
            p_values = [0, 6, 4, 2]
        evaluate_scatter(data, classes, kmeans, k_values, p_values)
    elif num_algorithm == 2:
        visualize_km(data, classes)

    return tagged_data, classes


if __name__ == "__main__":
    num_dataset = input_dataset()
    num_algorithm = input_algorithm()
    tagged_data, classes = execute_algorithm(num_dataset, num_algorithm)
