import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
import random

seed = 42
np.random.seed(seed)
random.seed(seed)

"""## Данные"""

data = pd.read_csv("gym_members_exercise_tracking.csv")
data.head()

data.Fat_Percentage.hist()

label_encoder = LabelEncoder()

data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Workout_Type'] = label_encoder.fit_transform(data['Workout_Type'])
data.head()

"""## Простой таргет"""

X, y = data.drop(columns=['Workout_Type']), ((data['Workout_Type'] < 2) - 0.5) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""### Дерево"""

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

y_pred = tree.predict(X_train)
print(f"{accuracy_score(y_train, y_train):.4f}")

y_pred = tree.predict(X_test)
print(f"{accuracy_score(y_test, y_pred):.4f}")

print(classification_report(y_test, y_pred))

"""### Ручной бустинг"""

class AdaptiveBoost:
    def __init__(self, n_estimators=50, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=self.max_depth, random_state=0)
            stump.fit(X, y, sample_weight=w)
            y_pred = stump.predict(X)
            err = w[(y_pred != y)].sum()
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            w *= np.exp(-alpha * y * y_pred)
            w /= w.sum()
            self.models.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        clf_preds = [alpha * model.predict(X) for alpha, model in zip(self.alphas, self.models)]
        y_pred = np.sign(sum(clf_preds))
        return y_pred

adaptive_boosting = AdaptiveBoost(25)
adaptive_boosting.fit(X_train, y_train)

y_pred = adaptive_boosting.predict(X_test)
accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))

adaptive_boosting = AdaptiveBoost(10)
adaptive_boosting.fit(X_train, y_train)
y_pred = adaptive_boosting.predict(X_test)
accuracy_score(y_test, y_pred)

adaptive_boosting = AdaptiveBoost(1)
adaptive_boosting.fit(X_train, y_train)
y_pred = adaptive_boosting.predict(X_test)
accuracy_score(y_test, y_pred)

"""### Библиотечный бустинг"""

base_estimator = DecisionTreeClassifier()
ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=25)

ada.fit(X_train, y_train)

y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))

"""## Сложный таргет"""

X, y = data.drop(columns=['Gender']), (data['Gender'] - 0.5) * 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""### Дерево"""

tree = DecisionTreeClassifier(max_depth=1)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print(f"{accuracy_score(y_test, y_pred):.4f}")

print(classification_report(y_test, y_pred))

"""### Ручной бустинг"""

adaptive_boosting = AdaptiveBoost(25, max_depth=1)
adaptive_boosting.fit(X_train, y_train)
y_pred = adaptive_boosting.predict(X_test)
accuracy_score(y_test, y_pred)

"""### Библиотечный бустинг"""

base_estimator = DecisionTreeClassifier(max_depth=1)
ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=25)

ada.fit(X_train, y_train)

y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))