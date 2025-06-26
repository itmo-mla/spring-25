import pandas as pd
import seaborn as sns
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from collections import Counter

df = sns.load_dataset("titanic")
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].dropna()
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

X = df.drop("survived", axis=1).values
y = df["survived"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def get_random_guess_accuracy(y_sample):
    counter = Counter(y_sample)
    total = len(y_sample)
    probs = [count / total for count in counter.values()]
    return sum(p ** 2 for p in probs)  # СЛУЧАЙНОЕ УГАДЫВАНИЕ

class CustomBaggingClassifier:
    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0, random_state=None, margin=0.05):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.margin = margin
        self.models = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.models = []
        n_samples = int(self.max_samples * len(X))

        while len(self.models) < self.n_estimators:
            indices = np.random.choice(len(X), n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            rand_acc = get_random_guess_accuracy(y_sample)
            threshold = rand_acc + self.margin

            model = self._clone_estimator()
            model.fit(X_sample, y_sample)
            y_pred_sample = model.predict(X_sample)
            acc = accuracy_score(y_sample, y_pred_sample)

            if acc >= threshold:
                self.models.append(model)
            else:
                print(f"[Отброшено] acc = {acc:.2f} < threshold = {threshold:.2f} (rand = {rand_acc:.2f})")

    def predict(self, X):
        if not self.models:
            raise ValueError("Ансамбль не обучен. Вызовите .fit() перед predict().")
        predictions = np.array([model.predict(X) for model in self.models])
        return np.round(predictions.mean(axis=0)).astype(int)

    def _clone_estimator(self):
        klass = type(self.base_estimator)
        params = self.base_estimator.get_params(deep=True)
        return klass(**params)

bagging = CustomBaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=50,
    max_samples=1.0,
    random_state=42,
    margin=0.05  # Константа для случайного угадывания
)

start_time = time.time()
bagging.fit(X_train, y_train)
train_time = time.time() - start_time

y_pred = bagging.predict(X_test)
acc = accuracy_score(y_test, y_pred)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = CustomBaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=50,
        max_samples=1.0,
        random_state=42,
        margin=0.05
    )
    model.fit(X_tr, y_tr)
    y_pred_val = model.predict(X_val)
    score = accuracy_score(y_val, y_pred_val)
    cv_scores.append(score)

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score

sk_bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
start_time = time.time()
sk_bagging.fit(X_train, y_train)
sk_train_time = time.time() - start_time
sk_y_pred = sk_bagging.predict(X_test)
sk_acc = accuracy_score(y_test, sk_y_pred)
sk_cv_scores = cross_val_score(sk_bagging, X, y, cv=5)

print("\nКастомная реализация")
print(f"Время обучения: {train_time:.4f} сек")
print(f"Точность на тесте: {acc:.4f}")
print(f"Средняя точность по кросс-валидации: {np.mean(cv_scores):.4f}")

print("\nРеализация scikit-learn")
print(f"Время обучения: {sk_train_time:.4f} сек")
print(f"Точность на тесте: {sk_acc:.4f}")
print(f"Средняя точность по кросс-валидации: {np.mean(sk_cv_scores):.4f}")
