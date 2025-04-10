import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from enum import Enum
from scipy.special import expit


class TaskType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class CustomGradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, task_type=TaskType.REGRESSION):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.task_type = task_type
        self.models = []
        self.f0 = None

    def fit(self, X, y):
        if self.task_type == TaskType.CLASSIFICATION:
            self.f0 = np.log(np.mean(y) / (1 - np.mean(y)))
            residuals = y - expit(self.f0)

            for _ in range(self.n_estimators):
                model = DecisionTreeClassifier(max_depth=self.max_depth)
                model.fit(X, residuals > 0)
                predictions = model.predict(X)
                residuals -= self.learning_rate * predictions
                self.models.append(model)

        elif self.task_type == TaskType.REGRESSION:
            self.f0 = np.mean(y)
            residuals = y - self.f0
            for _ in range(self.n_estimators):
                model = DecisionTreeRegressor(max_depth=self.max_depth)
                model.fit(X, residuals)
                predictions = model.predict(X)
                residuals -= self.learning_rate * predictions
                self.models.append(model)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def predict(self, X):
        if self.f0 is None:
            raise ValueError("Model not fitted yet.")

        if self.task_type == TaskType.CLASSIFICATION:
            logit_predictions = np.full(X.shape[0], self.f0)
            for model in self.models:
                logit_predictions += self.learning_rate * model.predict(X)
            return (expit(logit_predictions) > 0.5).astype(int)

        elif self.task_type == TaskType.REGRESSION:
            preds = np.full(X.shape[0], self.f0)
            for model in self.models:
                preds += self.learning_rate * model.predict(X)
            return preds