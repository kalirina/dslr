*This project has been created as part of the 42 curriculum by enrmarti and irkalini.*

# dslr

## Description

The goal of this project is to build a logistic regression model from scratch in Python, without using machine learning libraries. The implementation focuses on understanding how classification works, including gradient descent optimization and probability estimation using the sigmoid function.

---

## Instructions

Create virtual environment:

```sh
make
```

Activate virtual environment:

```sh
source .venv/bin/activate
```

Data Analysis:

```sh
python src/describe.py datasets/dataset_train.csv
```

Training the model:

```sh
python src/logreg_train.py datasets/dataset_train.csv
```

Predicting:

```sh
python src/logreg_predict.py datasets/dataset_test.csv
```

Clean the results (houses.csv, model.json):

```sh
make clean
```

Clean virtual environment:

```sh
make fclean
```

---

## Resources

### Technologies

* Python 3
* numpy
* pandas

### Articles & Tutorials

* [Multiclass logistic regression from scratch by GeeksforGeeks](https://medium.com/data-science/multiclass-logistic-regression-from-scratch-9cc0007da372)
* [Stochastic Gradient Descent by GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/ml-stochastic-gradient-descent-sgd/)

### AI Usage

AI tools were used as learning assistants to:

* clarify machine learning concepts;
* understand gradient descent implementation;
* debug matrix operations;
* improve documentation quality.

All code was written, understood, tested, and validated by the author.
