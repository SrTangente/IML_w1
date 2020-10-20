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
adult_df = adult_df.drop("class",axis=1)
adult_df = pd.get_dummies(adult_df)


waveform_df = pd.DataFrame(waveform_data)
waveform_df = waveform_df.drop("class",axis=1)

cn4_encoder = preprocessing.LabelEncoder()
cn4_encoder.fit_transform([b'x', b'b', b'o'])

cn4_proc_data = []
[cn4_proc_data.append(cn4_encoder.transform(x.tolist()[0:-1])) for x in cn4_data]
cn4_proc_data = [[cell-1 for cell in row] for row in cn4_proc_data]
