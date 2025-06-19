import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import time

iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

class CustomNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}

        for cls in self.classes:
            X_c = X[y == cls]
            self.parameters[cls] = {
                "mean": X_c.mean(axis=0),
                "var": X_c.var(axis=0) + 1e-9,
                "prior": X_c.shape[0] / X.shape[0]
            }

    def _pdf(self, class_idx, x):
        mean = self.parameters[class_idx]["mean"]
        var = self.parameters[class_idx]["var"]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_single(self, x):
        posteriors = []
        for cls in self.classes:
            prior = np.log(self.parameters[cls]["prior"])
            class_conditional = np.sum(np.log(self._pdf(cls, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

def cross_val_evaluation(model, X, y, cv=5, title=""):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    accuracies = []
    times = []
    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end = time.time()

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        times.append(end - start)
        all_true.extend(y_test)
        all_pred.extend(y_pred)

        print(f"[{title}] Fold {fold} Accuracy: {acc:.4f}")

    avg_acc = np.mean(accuracies)
    avg_time = np.mean(times)
    print(f"\n[{title}] Avg Accuracy: {avg_acc:.4f}, Avg Time: {avg_time:.6f} seconds")

    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{title} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return avg_acc, avg_time

print("Кастомная реализация")
custom_nb = CustomNaiveBayes()
custom_acc, custom_time = cross_val_evaluation(custom_nb, X, y, cv=5, title="Custom Naive Bayes")

print("\nРеализация scikit-learn")
sklearn_nb = GaussianNB()
sklearn_acc, sklearn_time = cross_val_evaluation(sklearn_nb, X, y, cv=5, title="Sklearn Naive Bayes")

print("\nСравнение")
print(f"Custom NB - Accuracy: {custom_acc:.4f}, Time: {custom_time:.6f}")
print(f"Sklearn NB - Accuracy: {sklearn_acc:.4f}, Time: {sklearn_time:.6f}")
