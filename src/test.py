from sklearn.metrics import accuracy_score
import pandas as pd


def main():
    true_houses = pd.read_csv("true_houses.csv")
    y_true = true_houses["Hogwarts House"]

    predictions = pd.read_csv("houses.csv")
    y_pred = predictions["Hogwarts House"]

    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy * 100, "%")
    print("Mistake percentage:", (1 - accuracy_score(y_true, y_pred)) * 100, "%")


if __name__ == "__main__":
    main()
