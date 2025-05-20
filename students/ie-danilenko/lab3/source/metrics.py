from model import NaiveBayes, KFold
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import accuracy_score
from read import read_heart
from sklearn.model_selection import train_test_split
from time import time

if __name__ == "__main__":
    X, y = read_heart("data/heart.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        NaiveBayes(distribution='gaussian'),
        GaussianNB()
    ]

    fold = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    times = []

    for model in models:
        start_time = time()
        metrics.append(fold.cros_valid(model, X_train, y_train, accuracy_score))
        times.append(time() - start_time)

    print(f"My time: {times[0]}")
    print(f"My metrics: {np.mean(metrics[0])}\n")

    print(f"Sklearn time: {times[1]}")
    print(f"Sklearn metrics: {np.mean(metrics[1])}") 