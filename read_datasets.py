from scipy.io import arff
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn import preprocessing
from CompareDBSCAN import compareDBSCAN_alg, compareDBSCAN_metric


def read_waveform():
    waveform_data, waveform_meta = arff.loadarff('./datasets/waveform.arff')
    waveform_df = pd.DataFrame(waveform_data)
    # Drop 'class' attribute, as it is the real label
    classes = waveform_df["class"]
    waveform_df = waveform_df.drop("class", axis=1)
    # No missing values, no step needed
    # Normalize the data
    scaler = preprocessing.MinMaxScaler()
    waveform_df_scaled = scaler.fit_transform(waveform_df.values)
    waveform_df = pd.DataFrame(waveform_df_scaled)
    return waveform_df.to_numpy(), classes


def read_adult():
    adult_data, adult_meta = arff.loadarff('./datasets/adult.arff')
    adult_df = pd.DataFrame(adult_data)
    # Replace missing values
    adult_df = adult_df.replace(b"?", np.nan)
    # Handle missing values (for now, just delete the row with the missing value)
    # The index can be reset after dropna() by calling reset_index(drop=True)
    # The index reset is omitted for now in order not to lose track of the missing values dropped
    adult_df = adult_df.dropna()
    # Drop 'class' attribute, as it is the real label
    classes = adult_df["class"]
    adult_df = adult_df.drop("class", axis=1)
    # One-hot encoding the categorical attributes
    adult_df = pd.get_dummies(adult_df)
    # Normalize the data
    scaler = preprocessing.MinMaxScaler()
    adult_df_scaled = scaler.fit_transform(adult_df.values)
    adult_df = pd.DataFrame(adult_df_scaled)
    return adult_df.to_numpy(), classes


def read_cn4():
    cn4_data, cn4_meta = arff.loadarff('./datasets/connect-4.arff')
    cn4_df = pd.DataFrame(cn4_data)
    # Drop 'class' attribute, as it is the real label
    classes = cn4_df["class"]
    cn4_df = cn4_df.drop("class", axis=1)
    # No missing values, no step needed
    # One-hot encoding the categorical attributes
    cn4_df = pd.get_dummies(cn4_df)
    # We do not need to normalize the data since all values are zeroes and ones
    return cn4_df.to_numpy(), classes
