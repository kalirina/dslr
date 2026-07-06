import numpy as np
import pandas as pd
import json
import sys
import os
import csv


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def extract(path, model):
    df = pd.read_csv(path)
    X = df[model["features"]].astype(float)
    # for feature in model["features"]:
    #     mean = model["mean"][feature]
    #     std = model["std"][feature]
    #     if std != 0:
    #         X[feature] = (X[feature] - mean) / std
    #     else:
    #         X[feature] = X[feature] - mean
    X = X.to_numpy()
    bias = np.ones((X.shape[0], 1))
    X = np.hstack((bias, X))
    return X


def predict(features, thetas):
    probabilities = {}
    for house, theta in thetas.items():
        z = np.dot(features, theta)
        probabilities[house] = sigmoid(z)
    return max(probabilities, key=probabilities.get)


def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        return 1
    dataset = sys.argv[1]
    if not os.path.exists(dataset):
        print("Wrong data set file")
        return 1
    try:
        with open("data/thetas.json", "r") as file:
            model = json.load(file)
    except FileNotFoundError:
        print("Train the model first")
        return 1
    data = extract(dataset, model)
    with open("houses.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Hogwarts House"])
        for index, student in enumerate(data):
            house = predict(student, model["thetas"])
            writer.writerow([index, house])


if __name__ == "__main__":
    main()
