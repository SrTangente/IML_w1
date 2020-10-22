from scipy.io import arff
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn import preprocessing
from CompareDBSCAN import compareDBSCAN_alg, compareDBSCAN_metric

adult_data, adult_meta = arff.loadarff('./datasets/adult.arff')
waveform_data, waveform_meta = arff.loadarff('./datasets/waveform.arff')
cn4_data, cn4_meta = arff.loadarff('./datasets/connect-4.arff')

adult_df = pd.DataFrame(adult_data)
# Drop 'class' attribute, as it is the real label
adult_df = adult_df.drop("class",axis=1)
# Replace missing values
adult_df = adult_df.replace(b"?",np.nan)
# One-hot encoding the categorical attributes
adult_df = pd.get_dummies(adult_df)
# Normalize the data
scaler = preprocessing.MinMaxScaler()
adult_df_scaled = scaler.fit_transform(adult_df.values)
adult_df = pd.DataFrame(adult_df_scaled)


waveform_df = pd.DataFrame(waveform_data)
# Drop 'class' attribute, as it is the real label
waveform_df = waveform_df.drop("class",axis=1)
# No missing values, no step needed
# Normalize the data
waveform_df_scaled = scaler.fit_transform(waveform_df.values)
waveform_df = pd.DataFrame(waveform_df_scaled)


cn4_df = pd.DataFrame(cn4_data)
# Drop 'class' attribute, as it is the real label
cn4_df = cn4_df.drop("class",axis=1)
# No missing values, no step needed
# One-hot encoding the categorical attributes
cn4_df = pd.get_dummies(cn4_df)
# We do not need to normalize the data since all values are zeroes and ones
