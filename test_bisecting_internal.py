from read_datasets import read_adult, read_cn4, read_waveform
from evaluate import evaluate_internal
from bisecting_k_means import bisecting_kmeans

#data, classes = read_waveform()
data, classes = read_cn4()
#data, classes = read_adult()

evaluate_internal(data, bisecting_kmeans)