from utils import sigmoid, extract
from loss_plot import plot_loss
import numpy as np
import json
import pandas as pd
import os
import sys


def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        return 1
    dataset = sys.argv[1]
    if not dataset.lower().endswith(".csv"):
        print("Dataset file must have a .csv extension")
        return 1
    if not os.path.exists(dataset):
        print("Dataset file doesn't exist")
        return 1
    if os.path.getsize(dataset) == 0:
        print("Dataset file is empty")
        return 1

    learning_rate = 0.01
    epochs = 1000
    houses = ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]

    if not os.path.exists("model.json"):
        print("Need to preprocess data fisrt")
        return 1

    with open("model.json", "r") as file:
        model = json.load(file)

    X = extract(dataset, model)

    df = pd.read_csv(dataset)
    y_houses = df["Hogwarts House"].values

    if "thetas" not in model:
        model["thetas"] = {}

    loss_history = {}

    for house in houses:
        if house not in model["thetas"]:
            model["thetas"][house] = [0.0] * X.shape[1]

        theta = np.array(model["thetas"][house], dtype=float)
        y = (y_houses == house).astype(int)
        m = len(y)
        loss_history[house] = []
        batch_size = 32

        for _ in range(epochs):
            indexes = np.random.permutation(m)
            for start in range(0, m, batch_size):
                batch = indexes[start:start + batch_size]
                X_batch = X[batch]
                y_batch = y[batch]

                predictions = sigmoid(np.dot(X_batch, theta))
                error = predictions - y_batch
                gradient = np.dot(X_batch.T, error) / len(batch)
                theta -= learning_rate * gradient

            # for plotting loss
            predictions = sigmoid(np.dot(X, theta))
            loss = -np.mean(y * np.log(predictions) + (1-y) * np.log(1-predictions))
            loss_history[house].append(loss)

        model["thetas"][house] = theta.tolist()

    plot_loss(loss_history, "Mini_batch_GD")

    with open("model.json", "w") as file:
        json.dump(model, file, indent=4)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
