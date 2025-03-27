import numpy as np

from sklearn import clone
from sklearn.model_selection import cross_val_score


class BaggingClassifier:
    def __init__(self, estimator, n_estimators=10, random_state=None, min_score_to_add=0.33):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_score_to_add = min_score_to_add
        self.models = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            other_indices = np.array(list(set(range(len(X))) - set(indices)))

            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            X_test = X[other_indices]
            y_test = y[other_indices]

            model = clone(self.estimator)
            # Если модель плохая, не добавляем ее в ансамбль
            while True:
                model.fit(X_bootstrap, y_bootstrap)
                score = cross_val_score(model, X_test, y_test, scoring='accuracy')
                if score.mean() > 1/len(np.unique(y)):
                    break
                model = clone(self.estimator)

            self.models.append(model)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])

        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions,
        )

        return final_predictions

    # Needed for sklearn cross-validation
    def get_params(self, *args, **kwargs):
        return dict(
            estimator=self.estimator,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )