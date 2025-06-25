import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import time
from sklearn.datasets import load_breast_cancer


class CustomNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        for cls in self.classes:
            X_cls = X[y == cls]
            self.mean[cls] = np.mean(X_cls, axis=0)
            self.var[cls] = np.var(X_cls, axis=0) + 1e-9
            self.priors[cls] = X_cls.shape[0] / n_samples

    def _gaussian(self, x, mean, var):
        exponent = -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)
        return exponent

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.priors[cls])
                likelihood = self._gaussian(x, self.mean[cls], self.var[cls])
                posterior = prior + likelihood
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)


def custom_cross_validation(model, X, y, cv=5):
    n_samples = X.shape[0]
    fold_size = n_samples // cv
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    scores = []

    for fold in range(cv):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < cv - 1 else n_samples
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

    return np.array(scores)


data = load_breast_cancer()
X, y = data.data, data.target
print(X)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("=== Самописный Naive Bayes ===")
start_time_custom = time.time()
custom_nb = CustomNaiveBayes()
custom_nb.fit(X_train, y_train)
custom_training_time = time.time() - start_time_custom


y_pred_custom = custom_nb.predict(X_test)
custom_accuracy = accuracy_score(y_test, y_pred_custom)


cv_scores_custom = custom_cross_validation(custom_nb, X, y, cv=5)
print(f"Accuracy: {custom_accuracy:.4f}")
print(f"Training time: {custom_training_time:.4f} seconds")
print(f"Custom cross-validation scores: {cv_scores_custom}")
print(
    f"Mean CV score: {cv_scores_custom.mean():.4f} (±{cv_scores_custom.std() * 2:.4f})"
)


print("\n=== Библиотечный GaussianNB ===")
start_time_lib = time.time()
lib_nb = GaussianNB()
lib_nb.fit(X_train, y_train)
lib_training_time = time.time() - start_time_lib


y_pred_lib = lib_nb.predict(X_test)
lib_accuracy = accuracy_score(y_test, y_pred_lib)


cv_scores_lib = custom_cross_validation(lib_nb, X, y, cv=5)
print(f"Accuracy: {lib_accuracy:.4f}")
print(f"Training time: {lib_training_time:.4f} seconds")
print(f"Custom cross-validation scores: {cv_scores_lib}")
print(f"Mean CV score: {cv_scores_lib.mean():.4f} (±{cv_scores_lib.std() * 2:.4f})")
