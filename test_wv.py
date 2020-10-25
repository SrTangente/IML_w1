from scipy.io import arff
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn import preprocessing
from CompareDBSCAN import compareDBSCAN_alg, compareDBSCAN_metric

waveform_data, waveform_meta = arff.loadarff('./datasets/waveform.arff')
scaler = preprocessing.MinMaxScaler()


waveform_df = pd.DataFrame(waveform_data)
# Drop 'class' attribute, as it is the real label
waveform_df = waveform_df.drop("class", axis=1)
# No missing values, no step needed
# Normalize the data
waveform_df_scaled = scaler.fit_transform(waveform_df.values)
waveform_df = pd.DataFrame(waveform_df_scaled)

waveform_np = waveform_df.to_numpy()
compareDBSCAN_alg(waveform_np, eps=0.41)
compareDBSCAN_metric(waveform_np, 0.41)
