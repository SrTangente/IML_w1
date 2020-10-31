from k_means import kmeans
import read_datasets
from evaluate import evaluate_external, evaluate_clustering
from sklearn.cluster import k_means
from sklearn import metrics

data, classes = read_datasets.read_waveform()
#data, classes = read_datasets.read_cn4()
#data, classes = read_datasets.read_adult()

evaluate_external(data, classes, kmeans)