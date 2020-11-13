from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


def read_waveform():
    waveform_data, waveform_meta = arff.loadarff('./datasets/waveform.arff')
    waveform_df = pd.DataFrame(waveform_data)
    # Drop 'class' attribute from the training, as we are using unsupervised learning
    classes = waveform_df["class"]
    waveform_df = waveform_df.drop("class", axis=1)
    # No missing values, no step needed
    # Normalize the data
    scaler = preprocessing.MinMaxScaler()
    waveform_df_scaled = scaler.fit_transform(waveform_df.values)
    waveform_df = pd.DataFrame(waveform_df_scaled)
    return waveform_df.to_numpy(), classes.to_numpy(dtype=np.float)


def read_adult():
    adult_data, adult_meta = arff.loadarff('./datasets/adult.arff')
    adult_df = pd.DataFrame(adult_data)
    # Replace missing values
    adult_df = adult_df.replace(b"?", np.nan)
    # Handle missing values (for now, just delete the row with the missing value)
    # The index can be reset after dropna() by calling reset_index(drop=True)
    # The index reset is omitted for now in order not to lose track of the missing values dropped
    adult_df = adult_df.dropna()
    # Drop 'class' attribute from the training, as we are using unsupervised learning
    classes = adult_df["class"]
    adult_df = adult_df.drop("class", axis=1)
    # Encode classes [b'<=50K', b'>50K'] to numbers
    enc = LabelEncoder()
    classes = enc.fit_transform(classes).astype(float)
    # One-hot encoding the categorical attributes
    adult_df = pd.get_dummies(adult_df)
    # Normalize the data
    scaler = preprocessing.MinMaxScaler()
    adult_df_scaled = scaler.fit_transform(adult_df.values)
    adult_df = pd.DataFrame(adult_df_scaled)
    return adult_df.to_numpy(), classes


def read_vowel():
    vowel_data, vowel_meta = arff.loadarff('./datasets/vowel.arff')
    vowel_df = pd.DataFrame(vowel_data)
    # Drop 'class' attribute from the training, as we are using unsupervised learning
    classes = vowel_df["Class"]
    vowel_df = vowel_df.drop("Class", axis=1)
    # Drop "train or test" and "speaker" columns
    vowel_df = vowel_df.drop("Train_or_Test", axis=1)
    vowel_df = vowel_df.drop("Speaker_Number", axis=1)
    # Encode classes to numbers
    enc = LabelEncoder()
    classes = enc.fit_transform(classes).astype(float)
    # One-hot encoding the categorical attributes
    vowel_df = pd.get_dummies(vowel_df)
    # Normalize the data
    scaler = preprocessing.MinMaxScaler()
    vowel_df_scaled = scaler.fit_transform(vowel_df.values)
    vowel_df = pd.DataFrame(vowel_df_scaled)
    return vowel_df.to_numpy(), classes

def read_cn4():
    cn4_data, cn4_meta = arff.loadarff('./datasets/connect-4.arff')
    cn4_df = pd.DataFrame(cn4_data)
    # Drop 'class' attribute from the training, as we are using unsupervised learning
    classes = cn4_df["class"]
    cn4_df = cn4_df.drop("class", axis=1)
    # Encode classes [b'draw', b'loss', b'win'] to numbers
    enc = LabelEncoder()
    classes = enc.fit_transform(classes).astype(float)
    # No missing values, no step needed
    # One-hot encoding the categorical attributes
    cn4_df = pd.get_dummies(cn4_df)
    # We do not need to normalize the data since all values are zeroes and ones
    return cn4_df.to_numpy(), classes
