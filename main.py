from read_datasets import read_cn4, read_adult, read_waveform
from k_means import kmeans
from bisecting_k_means import bisecting_kmeans
from KHM import KHM
from evaluate import evaluate_clustering, evaluate_DBSCAN
from CompareDBSCAN import compareDBSCAN_alg

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

def evaluate_algorithm(num_algorithm, tagged_data, classes):
    if num_algorithm == 1:
        # TODO: this is provisional. Substitute for evaluate_DBSCAN() once
        #  DBSCAN is available to be selected.
        evaluate_clustering(tagged_data, classes)
    else:
        evaluate_clustering(tagged_data, classes)

def execute_algorithm(num_algorithm, data, k):
    if num_algorithm == 1:
        # TODO: this is provisional. Replace with the best DBSCAN obtained
        return bisecting_kmeans(data, k)
    elif num_algorithm == 2:
        return kmeans(data, k)
    elif num_algorithm == 3:
        return bisecting_kmeans(data, k)
    elif num_algorithm == 4:
        # TODO: Call KHM with best parameters obtained
        return KHM(data, k, 4, 100, 0.01)
    else:
        # TODO: Substitute for Fuzzy C-Means
        return bisecting_kmeans(data, k)

if __name__ == "__main__":

    num_dataset = input_dataset()
    num_algorithm = input_algorithm()
    (data, classes), k = read_dataset(num_dataset)
    tagged_data = execute_algorithm(num_algorithm, data, k)
    evaluate_algorithm(num_algorithm, tagged_data, classes)
