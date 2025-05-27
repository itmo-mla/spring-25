import numpy as np
from sklearn.tree import DecisionTreeRegressor


class CustomGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.estimators = []
        self._rng = np.random.RandomState(random_state)

        self._training_loss = []
        self.y_shape = None

    def _subsample_data(self, X, y):
        if self.subsample < 1.0:
            n_samples = X.shape[0]
            subsample_size = int(n_samples * self.subsample)
            indices = self._rng.choice(n_samples, subsample_size, replace=False)
            return X[indices], y[indices]
        return X, y

    def fit(self, X, y):
        pred = np.zeros_like(y, dtype=np.float64)
        residuals = y - pred
        self.y_shape = y.shape[1:]
        for i in range(self.n_estimators):
            X_subsampled, residuals_subsampled = self._subsample_data(X, residuals)

            estimator = DecisionTreeRegressor(max_depth=self.max_depth,
                                              random_state=self._rng)
            estimator.fit(X_subsampled, residuals_subsampled)

            update = estimator.predict(X)
            pred += self.learning_rate * update.reshape(-1, *self.y_shape)

            self.estimators.append(estimator)

            residuals = y - pred
            self._training_loss.append(residuals.mean())

        return self

    def predict(self, X):
        pred = np.zeros(X.shape[0], dtype=np.float64)
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)
        return pred.reshape(-1, *self.y_shape)

    def get_history(self):
        return self._training_loss


