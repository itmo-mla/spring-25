from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from graph import decision_boundary_plot
import time
from sklearn.naive_bayes import GaussianNB

from Bayes import GaussianNaiveBayes
from CrossValidator import CrossValidation

# Загрузка данных
data = pd.read_csv('glass.csv')
X, y = data.iloc[:, :-1], data.iloc[:, -1]
y = pd.Series(LabelEncoder().fit_transform(y))


def time_of_method(func):
    def wrapper():
        start = time.time()
        func()
        end = time.time()
        delta = end - start
        print(f'{delta}')
    return wrapper


@time_of_method
def hand_method():
    model = GaussianNaiveBayes()
    cross_validator = CrossValidation(n_splits=5)

    acc = cross_validator.eval(X, y, model, accuracy_score)
    f1 = cross_validator.eval(X, y, model, f1_score)

    # Обучение бэггинга
    print(f"Accuracy hand method: {acc}")
    print(f"F1 hand method: {f1}")


@time_of_method
def sklearn_method():
    model = GaussianNB()
    cross_validator = CrossValidation(n_splits=5)

    acc = cross_validator.eval(X, y, model, accuracy_score)
    f1 = cross_validator.eval(X, y, model, f1_score)

    print(f"Sklearn accuracy: {acc}")
    print(f"Sklearn f1: {f1}")


if __name__ == "__main__":
    hand_method()
    print()

    sklearn_method()
    decision_boundary_plot(X, y, X, y, GaussianNB(), [2, 3])

