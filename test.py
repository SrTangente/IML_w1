from scipy.io import arff
import numpy as np
from k_means import kmeans

waveform_data, waveform_meta = arff.loadarff('./datasets/waveform.arff')

waveform_final_data = np.zeros([len(waveform_data), len(waveform_data[0])-1])

for i in range(len(waveform_data)):
    for j in range(len(waveform_data[0])-1):
        waveform_final_data[i, j] = waveform_data[i][j]

tagged_data = kmeans(waveform_final_data, 3)

[print(tagged_data[i, -1], waveform_data[i][-1]) for i in range(50)]
