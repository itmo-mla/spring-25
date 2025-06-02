import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBMRegressor:
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3, random_state=12345):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.initial_leaf = None

    def fit(self, X, y):
        self.initial_leaf = 0
        predictions = np.zeros(len(y)) + self.initial_leaf

        for _ in range(self.n_estimators):
            residuals = 2 * (y - predictions)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, residuals)

            predictions += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, samples):
        predictions = np.zeros(len(samples)) + self.initial_leaf

        for i in range(self.n_estimators):
            predictions += self.learning_rate * self.trees[i].predict(samples)

        return predictions


class CrossValidation:
    def __init__(self, n_splits, random_state=12, shuffle=True):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

    def split(self, X):
        X = np.array(X)
        n_samples = len(X)

        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.seed(self.random_state)
            indices = np.random.permutation(indices)

        fold_size = n_samples // self.n_splits
        folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(self.n_splits)]

        remainder = n_samples % self.n_splits
        if remainder != 0:
            folds[-1] = np.concatenate([folds[-1], indices[-remainder:]])

        for fold_idx in range(self.n_splits):
            val_indices = folds[fold_idx]
            train_indices = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx])
            yield train_indices, val_indices

    def eval(self, X, y, model, metric):
        metrics = []
        X, y = np.array(X), np.array(y)

        for train_idx, val_idx in self.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, y_valid = X[val_idx], y[val_idx]

            model.fit(X_train, y_train)
            pred = model.predict(X_valid)

            metrics.append(metric(pred, y_valid))

        return np.array(metrics)
