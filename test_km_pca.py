from k_means import kmeans
import read_datasets
from evaluate import evaluate_external, evaluate_internal
from sklearn.cluster import k_means
from sklearn import metrics
from sklearn import decomposition

data, classes = read_datasets.read_vowel()
#data, classes = read_datasets.read_cn4()
#data, classes = read_datasets.read_adult()

print(data.shape[1])

values = []

for p in values:
    pca = decomposition.PCA(n_components=p)
    data = pca.fit_transform(data)
    evaluate_internal(data, kmeans)
