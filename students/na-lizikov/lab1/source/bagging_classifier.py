import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from typing import List
import pandas as pd

class ManualBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators: int = 10, max_samples: float = 1.0, min_oob_score: float = 0.5):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.min_oob_score = min_oob_score
        self.estimators_: List[DecisionTreeClassifier] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.estimators_ = []
        if not isinstance(X, pd.DataFrame):
             X = pd.DataFrame(X)

        n_samples = int(X.shape[0] * self.max_samples)
        all_indices = np.arange(X.shape[0])

        while len(self.estimators_) < self.n_estimators:
            bootstrap_indices = resample(all_indices, n_samples=n_samples, replace=True)
            X_bootstrap, y_bootstrap =  X.iloc[bootstrap_indices], y[bootstrap_indices]
            oob_indices = np.setdiff1d(all_indices, np.unique(bootstrap_indices))
            if len(oob_indices) == 0:
                continue
            X_oob, y_oob = X.iloc[oob_indices], y[oob_indices]

            estimator = DecisionTreeClassifier(random_state=None)
            estimator.fit(X_bootstrap, y_bootstrap)
            oob_score = estimator.score(X_oob, y_oob)
            if oob_score > self.min_oob_score:
                self.estimators_.append(estimator)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Предсказание класса для X с использованием голосования по большинству
        if not isinstance(X, pd.DataFrame):
             X = pd.DataFrame(X)
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
