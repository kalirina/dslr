import numpy as np
import json
import sys


def sigmoid(z):
    return 1 / (1 + np.exp(z))


def main():
    if sys.args != 3:
        return
    try:
        with open("data/thetas.json", "r") as read_file:
            thetas = json.load(read_file)
            return sigmoid(x.dot(thetas))
    except FileNotFoundError:
        return 0
    return sigmoid(theta x's)


if __name__ == "__main__":
    main()
