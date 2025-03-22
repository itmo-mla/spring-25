from sklearn.base import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

class RandomSubspaceMethod:
    def __init__(
        self,
        BaseEstimator=DecisionTreeClassifier,
        n_estimators=10,
        subspace_size=0.5,
        random_state=None,
    ):
        self.BaseEstimator = BaseEstimator
        self.n_estimators = n_estimators
        self.subspace_size = subspace_size
        self.random_state = random_state
        self.estimators = []
        self.feature_indices = []

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray):
        _n_samples, n_features = X.shape
        n_subspace_features = max(1, int(n_features * self.subspace_size))

        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            feature_indices = rng.choice(n_features, n_subspace_features, replace=False)
            self.feature_indices.append(feature_indices)
            X_subspace = X[:, feature_indices]

            accuracy = 0
            while accuracy < 0.5:
                estimator = self.BaseEstimator()
                estimator.fit(X_subspace, y)
                y_pred = estimator.predict(X_subspace)
                accuracy = accuracy_score(y, y_pred)
            else:
                self.estimators.append(estimator)
        return self

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.estimators)))

        for estimator_idx, (estimator, features) in enumerate(
            zip(self.estimators, self.feature_indices)
        ):
            predictions[:, estimator_idx] = estimator.predict(X[:, features])

        return np.array([np.bincount(row.astype(int)).argmax() for row in predictions])
