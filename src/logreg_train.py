from utils import sigmoid, extract
import numpy as np
import json
import pandas as pd


def main():
    learning_rate = 0.1
    epochs = 1000

    houses = ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]
    with open("data/thetas.json", "r") as file:
        model = json.load(file)
    X = extract("datasets/dataset_train.csv", model)
    # load full dataset again for labels
    df = pd.read_csv("datasets/dataset_train.csv")
    y_houses = df["Hogwarts House"].values

    for house in houses:
        y = (y_houses == house).astype(int)
        theta = np.array(model["thetas"][house])
        m = len(y)
        for epoch in range(epochs):
            predictions = sigmoid(np.dot(X, theta))
            error = predictions - y
            gradient = np.dot(X.T, error) / m
            theta -= learning_rate * gradient
    with open("data/thetas.json", "w") as file:
        json.dump(model, file)


if __name__ == "__main__":
    main()
