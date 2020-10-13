from scipy.io import arff
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn import preprocessing
from CompareDBSCAN import compareDBSCAN
zoo_data, zoo_meta = arff.loadarff('./datasets/zoo.arff')
waveform_data, waveform_meta = arff.loadarff('./datasets/waveform.arff')
cn4_data, cn4_meta = arff.loadarff('./datasets/connect-4.arff')

zoo_encoder = preprocessing.LabelEncoder()
zoo_encoder.fit([b'true', b'false'])
ohe = preprocessing.OneHotEncoder()

zoo_p_data = np.zeros([len(zoo_data), len(zoo_data[0])])

for i in range(len(zoo_data)):
    for j in range(1,13):
         zoo_p_data[i,j] = zoo_encoder.transform([zoo_data[i][j]])[0]
    # We have to scale this range to [0,1] but I don't understand the method xd
    zoo_p_data[i, 13] = zoo_data[i][13]
    for j in range(14,17):
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

compareDBSCAN(cn4_proc_data)
compareDBSCAN(waveform_final_data)
compareDBSCAN(zoo_final_data)