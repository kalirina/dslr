from utils import sigmoid, extract, preprocess_tmp
import numpy as np
import json
import pandas as pd
import os


def main():
    learning_rate = 0.1
    epochs = 1000
    houses = ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]

    if not os.path.exists("src/model.json"):
        print("Need to preprocess data fisrt")
        return 0

    #delete after:
    preprocess_tmp("datasets/dataset_train.csv")

    with open("src/model.json", "r") as file:
        model = json.load(file)

    X = extract("datasets/dataset_train.csv", model)

    df = pd.read_csv("datasets/dataset_train.csv")
    y_houses = df["Hogwarts House"].values

    if "thetas" not in model:
        model["thetas"] = {}

    for house in houses:
        if house not in model["thetas"]:
            model["thetas"][house] = [0.0] * X.shape[1]

        theta = np.array(model["thetas"][house], dtype=float)
        y = (y_houses == house).astype(int)
        m = len(y)

        for _ in range(epochs):
            predictions = sigmoid(np.dot(X, theta))
            error = predictions - y
            gradient = np.dot(X.T, error) / m
            theta -= learning_rate * gradient

        model["thetas"][house] = theta.tolist()

    with open("src/model.json", "w") as file:
        json.dump(model, file, indent=4)


if __name__ == "__main__":
    main()
