import numpy as np
import json
import sys
import os
import csv
from utils import sigmoid, extract


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

    try:
        with open("model.json", "r") as file:
            model = json.load(file)
    except FileNotFoundError:
        print("Train the model first")
        return 1
    data = extract(dataset, model)

    with open("houses.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Hogwarts House"])
        for index, student in enumerate(data):
            probabilities = {}
            for house, theta in model["thetas"].items():
                z = np.dot(student, theta)
                probabilities[house] = sigmoid(z)
            house = max(probabilities, key=probabilities.get)
            writer.writerow([index, house])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
