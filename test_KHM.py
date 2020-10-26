from KHM import KHM
import numpy as np
from read2pandas import read2pandas

ds=read2pandas('./datasets/waveform.arff')[0].drop('class',axis=1).to_numpy()
c=KHM(ds,5,2,1000,10e-12)