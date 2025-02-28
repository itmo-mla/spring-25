from joblib import Parallel, delayed

import numpy as np

from sklearn import clone


class BaggingClassifier:
    def __init__(
        self,
        estimator,
        n_estimators=10,
        n_jobs=8,
        min_quality_threshold=0.5,
        random_state=None,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.min_quality_threshold = min_quality_threshold
        self.models = []

    def _train_single_model(self, X, y):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        oob_indices = np.array(list(set(range(len(X))) - set(indices)))
        X_train, y_train = X[indices], y[indices]
        X_test, y_test = X[oob_indices], y[oob_indices]

        model = clone(self.estimator)
        model.fit(X_train, y_train)

        return model, np.mean(model.predict(X_test) == y_test)

    def train_single_model_until_good_quality(self, X, y):
        for _try in range(10):
            model, acc = self._train_single_model(X, y)
            if acc > self.min_quality_threshold:
                return model
        raise TimeoutError

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.models = list(Parallel(n_jobs=self.n_jobs)(
            delayed(self.train_single_model_until_good_quality)(X, y)
            for _ in range(self.n_estimators)
        ))

    def predict(self, X):
        # Получаем предсказания от каждой модели
        predictions = np.array([model.predict(X) for model in self.models])

        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions,
        )

        return final_predictions

    def get_params(self, *args, **kwargs):
        return dict(
            estimator=self.estimator,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
