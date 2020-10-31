from KHM import KHM
from k_means import kmeans
import read_datasets
import numpy as np
from evaluate import evaluate_external
from bisecting_k_means import bisecting_kmeans

data, classes = read_datasets.read_waveform()
#data, classes = read_datasets.read_cn4()
#data, classes = read_datasets.read_adult()

evaluate_external(data, classes, bisecting_kmeans)



