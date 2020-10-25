from scipy.io import arff
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn import preprocessing
from CompareDBSCAN import compareDBSCAN_alg, compareDBSCAN_metric
from sklearn import metrics
from k_means import kmeans

adult_data, adult_meta = arff.loadarff('./datasets/adult.arff')

adult_df = pd.DataFrame(adult_data)
# Drop 'class' attribute, as it is the real label
adult_df = adult_df.drop("class", axis=1)
# Replace missing values
adult_df = adult_df.replace(b"?", np.nan)
# Handle missing values (for now, just delete the row with the missing value)
# The index can be reset after dropna() by calling reset_index(drop=True)
# The index reset is omitted for now in order not to lose track of the missing values dropped
adult_df = adult_df.dropna()
# One-hot encoding the categorical attributes
adult_df = pd.get_dummies(adult_df)
# Normalize the data
scaler = preprocessing.MinMaxScaler()
adult_df_scaled = scaler.fit_transform(adult_df.values)
adult_df = pd.DataFrame(adult_df_scaled)
adult_np = adult_df.to_numpy()

#compareDBSCAN_alg(adult_np, eps=0.3)
#compareDBSCAN_metric(adult_np, eps=0.3)

print(kmeans(adult_np, )[:, -1])
