from aeon.datasets import load_from_tsf_file
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def read_tsf(path="cif_2016_dataset.ts", start=36, stop=46):
    data, meta = load_from_tsf_file(path)

    x = data.series_value[start:stop].to_numpy()
    return x

def split_data(values, test_split=12):
    x_train = []
    x_test = []

    # Split training and test sets
    for ts in values:
        x_train.append(ts[:len(ts) - test_split])
        x_test.append(ts[:test_split])

    return x_train, x_test


# Function to reshape the input data and scale it
def prepare_data(values):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values.reshape(-1, 1))

    # Create the time steps
    X, Y = [], []
    for i in range(len(scaled_values) - 1):
        X.append([i])
        Y.append(scaled_values[i + 1])
    return np.array(X), np.array(Y), scaler
