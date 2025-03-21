import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize

def sigmoid( x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def log_loss( y_true: np.ndarray, y_pred: np.ndarray):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def log_loss_gradient( y_true: np.ndarray, y_pred: np.ndarray):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))


class CustomGradientBoosting:
    def __init__(
        self,
        n_estimators=100,
        BaseAlgorithmType: type[DecisionTreeRegressor] = DecisionTreeRegressor,
        learning_rate=0.02,
        max_depth=5,
    ):
        self.n_estimators = n_estimators
        self.BaseAlgorithmType = BaseAlgorithmType
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y):
        self._base_algorithms = []
        self._alphas = []
        pred_y = np.zeros_like(y, dtype=np.float64)

        for t in range(0, self.n_estimators):
            base_algorithm = DecisionTreeRegressor(
                criterion="squared_error", max_depth=5
            )
            base_algorithm.fit(X, -log_loss_gradient(y, sigmoid(pred_y)))
            alpha = minimize(
                lambda alpha: log_loss(
                    y,
                    sigmoid(pred_y + alpha * base_algorithm.predict(X)),
                ),
                0.001,
                method="L-BFGS-B",
                bounds=[(0.001, 10)],
                options={"ftol": 1e-6, "gtol": 1e-6},
            ).x[0]

            pred_y += self.learning_rate * alpha * base_algorithm.predict(X)
            self._base_algorithms.append(base_algorithm)
            self._alphas.append(alpha)

    def predict_proba(self, X):
        predictions = np.zeros(len(X))
        for base_algorithm, alpha in zip(self._base_algorithms, self._alphas):
            predictions += self.learning_rate * alpha * base_algorithm.predict(X)
        return sigmoid(predictions)

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0.5, 1, 0)
