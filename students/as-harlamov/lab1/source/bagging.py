import numpy as np

from sklearn import clone


class BaggingClassifier:
    def __init__(self, estimator, n_estimators=10, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            model = clone(self.estimator)
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)

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
