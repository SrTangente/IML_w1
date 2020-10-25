from scipy.io import arff
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn import preprocessing
from CompareDBSCAN import compareDBSCAN_alg, compareDBSCAN_metric
from k_means import kmeans
from sklearn import metrics


cn4_data, cn4_meta = arff.loadarff('./datasets/connect-4.arff')


cn4_df = pd.DataFrame(cn4_data)
classes = cn4_df["class"]
# Drop 'class' attribute, as it is the real label
cn4_df = cn4_df.drop("class", axis=1)
# No missing values, no step needed
# One-hot encoding the categorical attributes
cn4_df = pd.get_dummies(cn4_df)
# We do not need to normalize the data since all values are zeroes and ones

cn4_np = cn4_df.to_numpy()


#compareDBSCAN_alg(cn4_np, eps=0.3)
#compareDBSCAN_metric(cn4_np, eps=0.3)

clustering = kmeans(cn4_np, 3)
labels = clustering[:, -1]
print(labels)

ars = metrics.adjusted_rand_score(classes, labels)
print(ars)
