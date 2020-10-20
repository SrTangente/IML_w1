from scipy.io import arff
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn import preprocessing
from CompareDBSCAN import compareDBSCAN_alg, compareDBSCAN_metric

adult_data, adult_meta = arff.loadarff('./datasets/adult.arff')
waveform_data, waveform_meta = arff.loadarff('./datasets/waveform.arff')
cn4_data, cn4_meta = arff.loadarff('./datasets/connect-4.arff')

adult_data_tmp = np.empty_like(adult_data)
for idx,row in enumerate(adult_data[:10]):
    row_tmp = ()
    for attr in row:
        try:
            decoded = attr.decode()
            row_tmp += (decoded,)
            print(row_tmp)
        except AttributeError:
            row_tmp += (attr,)
            print(row_tmp)
    adult_data_tmp[idx] = row_tmp
    print(adult_data_tmp[:10])



adult_data_pd = pd.DataFrame(adult_data_tmp)
ohe = preprocessing.OneHotEncoder()

adult_data_pd = ohe.fit_transform(adult_data_pd)
zoo_p_data = np.zeros([len(zoo_data), len(zoo_data[0])])

for i in range(len(zoo_data)):
    for j in range(1,13):
         zoo_p_data[i,j] = zoo_encoder.transform([zoo_data[i][j]])[0]
    # We have to scale this range to [0,1] but I don't understand the method xd
    zoo_p_data[i, 13] = zoo_data[i][13]
    for j in range(14, 17):
         zoo_p_data[i,j] = zoo_encoder.transform([zoo_data[i][j]])[0]

zoo_final_data = zoo_p_data[:, 1:-1]

waveform_final_data = np.zeros([len(waveform_data), len(waveform_data[0])-1])

for i in range(len(waveform_data)):
    for j in range(len(waveform_data[0])-1):
        waveform_final_data[i,j] = waveform_data[i][j]

cn4_encoder = preprocessing.LabelEncoder()
cn4_encoder.fit([b'x', b'b', b'o'])

cn4_proc_data = []
[cn4_proc_data.append(cn4_encoder.transform(x.tolist()[0:-1])) for x in cn4_data]