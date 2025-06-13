import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_data(filename):
    label = LabelEncoder()

    data = pd.read_csv(filename)
    data = data.dropna()
    data['Pop'] = label.fit_transform(data['Pop'])
    data['sex'] = label.fit_transform(data['sex'])
    y = data['age'].to_numpy()
    del data['age']
    X = data.to_numpy()
    return X, y


import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.01, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def _gradient_calc(self, y_true, y_pred):
        return -(y_true - y_pred)

    def _tree_creator(self, X, y):
        from sklearn.tree import DecisionTreeRegressor
        tree = DecisionTreeRegressor(max_depth=self.max_depth)
        tree.fit(X, y)
        return tree

    def fit(self, X, y):
        import numpy as np
        X = np.array(X)
        y = np.array(y)

        self.initial_prediction = np.mean(y)
        current_predictions = np.full(y.shape, self.initial_prediction)

        for i in range(self.n_estimators):
            gradients = self._gradient_calc(y, current_predictions)
            tree = self._tree_creator(X, gradients)
            tree_preds = tree.predict(X)
            current_predictions -= self.learning_rate * tree_preds
            self.trees.append(tree)

        return self

    def predict(self, X):
        import numpy as np
        X = np.array(X)

        predictions = np.full(X.shape[0], self.initial_prediction)

        for tree in self.trees:
            tree_preds = tree.predict(X)
            predictions -= self.learning_rate * tree_preds

        return predictions

class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def __split(self, X):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        fold_sizes = (n_samples // self.n_splits) * np.ones(self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, end = current, current + fold_size
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            yield train_indices, val_indices
            current = end

    def cross_valid(self, model, X, y, metrics):
        metric_scores = []

        for train_idx, val_idx in self.__split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            metric_scores.append(metrics(y_pred, y_val))

        return np.asarray(metric_scores)


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from time import time

X, y = read_data("possum.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=1000, max_depth=16, learning_rate=0.005)



fold = KFold(n_splits=5, shuffle=True, random_state=42)
start_time = time()
scores = fold.cross_valid(model, X_train, y_train, mean_squared_error)
print(f"Train time: {time() - start_time}")
print("Cross-validation MSE:", np.mean(scores))

from sklearn.ensemble import GradientBoostingRegressor as GBSK

X, y = read_data("possum.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    GradientBoostingRegressor(n_estimators=200, max_depth=32, learning_rate=0.001),
    GBSK(n_estimators=200, max_depth=32, learning_rate=0.001, criterion="squared_error")
]

fold = KFold()

metrics = []
times = []
for model in models:
    start_time = time()
    metrics.append(fold.cross_valid(model, X_train, y_train, mean_squared_error))
    times.append(time() - start_time)

print(f"{'-'*20}CUSTOM GB time: {times[0]}{'-'*20}")
print(f"{'-'*20}CUSTOM GB metrics: {np.mean(metrics[0])}{'-'*20}")
print(f"{'-'*20}")
print(f"{'-'*20}SKGB time: {times[1]}{'-'*20}")
print(f"{'-'*20}SKGB metrics: {np.mean(metrics[1])}{'-'*20}")