from utils import sigmoid
import numpy as np


def predict(X, theta):
    return sigmoid(np.dot(X, theta))


def main():
    for house in houses:
        for epoch in range(1000):
            predictions = predict(data)



if __name__ == "__main__":
    main()
