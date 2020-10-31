from CompareDBSCAN import compareDBSCAN_alg, compareDBSCAN_metric
from read_datasets import *
import numpy as np
import sklearn

data, classes = read_adult()
#data, classes = read_waveform()
#data, classes = read_cn4()

compareDBSCAN_metric(data, classes, 0.09, np.log(len(data)))
compareDBSCAN_alg(data, classes, 0.09, np.log(len(data)))
