from read_datasets import *
from evaluate import evaluate_internal
from k_means import kmeans

#data, classes = read_waveform()
data, classes = read_vowel()
#data, classes = read_adult()

evaluate_internal(data, kmeans)
