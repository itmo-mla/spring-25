import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import defaultdict

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

data.info()

X, y = data[['Gender', 'Workout_Type']], data['Experience_Level']
X['Workout_Type'] = X['Workout_Type'] < 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def ensure_numeric(array):
    return np.array(array, dtype=np.float64)

X_train = ensure_numeric(X_train)
X_test = ensure_numeric(X_test)

"""### Наивный Байес"""

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.means = {}
        self.variances = {}

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)

        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = X_cls.shape[0] / X.shape[0]
            self.means[cls] = X_cls.mean(axis=0)
            self.variances[cls] = X_cls.var(axis=0, ddof=1) + 1e-9

    def _gaussian_pdf(self, x, mean, var):
        return np.exp(-(x - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)

    def predict(self, X):
        X = np.array(X)
        predictions = []

        for sample in X:
            max_log_prob = -np.inf
            best_class = None

            for cls in self.classes:
                log_prob = np.log(self.class_priors[cls])
                log_prob += np.sum(np.log(self._gaussian_pdf(sample, self.means[cls], self.variances[cls])))

                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    best_class = cls

            predictions.append(best_class)

        return np.array(predictions)

nb = GaussianNaiveBayes()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

class BernoulliNaiveBayes:
    def init(self):
        self.classes = None
        self.class_priors = None
        self.bernoulli_probs = None
        self._is_fitted = False

    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_priors = (counts + self.alpha) / (len(y) + self.alpha * len(self.classes))

        n_features = X.shape[1]
        self.bernoulli_probs = defaultdict(dict)

        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            for feature in range(n_features):
                count_1 = np.sum(X_cls[:, feature]) + self.alpha
                total = X_cls.shape[0] + 2 * self.alpha
                self.bernoulli_probs[cls][feature] = count_1 / total

        self._is_fitted = True

    def predict(self, X):
        if not self._is_fitted:
            raise ValueError("Call fit() before predict()")

        log_probs = []
        for cls in self.classes:
            class_log_prob = np.log(self.class_priors[i])
            feature_log_probs = [
                np.where(X[:, i] == 1,
                         np.log(self.bernoulli_probs[cls][i]),
                         np.log(1 - self.bernoulli_probs[cls][i]))
                for i in range(X.shape[1])
            ]
            log_probs.append(class_log_prob + np.sum(feature_log_probs, axis=0))

        return self.classes[np.argmax(log_probs, axis=0)]

bnb = BernoulliNaiveBayes()
bnb.fit(X_train, y_train)

accuracy = accuracy_score(y_test, bnb.predict(X_test))
print(f"Accuracy: {accuracy:.2f}")

"""### Библиотечная версия"""

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))