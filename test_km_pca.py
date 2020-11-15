from k_means import kmeans
import read_datasets
import evaluate
from sklearn.cluster import k_means
from sklearn import metrics
from sklearn import decomposition


#data, classes = read_datasets.read_vowel()
#data, classes = read_datasets.read_waveform()
data, classes = read_datasets.read_adult()


k_values = [2, 3]
p_values = [0, 10, 5, 2]

evaluate.evaluate_scatter(data, classes, kmeans, k_values, p_values)
