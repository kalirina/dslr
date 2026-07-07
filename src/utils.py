import numpy as np
import pandas as pd
import json


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# SHOULD BE DONE IN PREPROCESSING PART
def preprocess_tmp(path):
    df = pd.read_csv(path)
    features = df.select_dtypes(include="number").columns.tolist()
    if "Index" in features:
        features.remove("Index")
    model = {"features": features}
    with open("src/model.json", "w") as file:
        json.dump(model, file, indent=4)


def extract(path, model):
    df = pd.read_csv(path)
    X = df[model["features"]].astype(float)
    X = X.to_numpy()
    bias = np.ones((X.shape[0], 1))
    X = np.hstack((bias, X))
    return X
