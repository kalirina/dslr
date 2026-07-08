import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def extract(path, model):
    df = pd.read_csv(path)
    if df.empty:
        print("Error: dataset is empty")
        return
    X = df[model["features"]].astype(float)
    for feature in model["features"]:
        mean = model["mean"][feature]
        std = model["std"][feature]
        # Fill missing values
        X[feature] = X[feature].fillna(mean)
        # Normalize
        if std != 0:
            X[feature] = (X[feature] - mean) / std
        else:
            X[feature] = X[feature] - mean
    X = X.to_numpy()
    bias = np.ones((X.shape[0], 1))
    X = np.hstack((bias, X))
    return X
