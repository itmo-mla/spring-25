import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed


class BaggingClassifier:
    def __init__(self, estimator=DecisionTreeClassifier(), n_estimators=10, accuracy_threshold=0.8, random_state=42, n_jobs=4):
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.data_random_seeds = []
        self.accuracy_threshold = accuracy_threshold
        self.n_jobs = n_jobs

        self.random_state = random_state

        self.estimators = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.estimators = list(Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_by_treshold)(X, y)
            for _ in range(self.n_estimators)
        ))

    def train_estimator(self, X, y):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        oob_indices = np.array(list(set(range(len(X))) - set(indices)))
        X_train, y_train = X[indices], y[indices]
        X_test, y_test = X[oob_indices], y[oob_indices]

        estimator = self.estimator
        estimator.fit(X_train, y_train)

        return estimator, accuracy_score(estimator.predict(X_test), y_test)

    def train_by_treshold(self, X, y):
        for _ in range(5):
            model, accuracy = self.train_estimator(X, y)
            if accuracy > self.accuracy_threshold:
                return model
        raise TimeoutError

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.estimators])
        final_prediction = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions,
        )
        return final_prediction

    def get_params(self, *args, **kwargs):
        return dict(
            estimator=self.estimator,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )