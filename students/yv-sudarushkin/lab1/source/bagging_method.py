import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class CustomBaggingClassification:
    def __init__(self, n_estimators=10, threshold_train_score=0.51, threshold_test_score=0.51):
        self.n_estimators = n_estimators

        self.estimators = []
        self.data_random_seeds = []

        self.threshold_train_score = threshold_train_score
        self.threshold_test_score = threshold_test_score

    def fit(self, X, y):
        self.estimators = []
        while len(self.estimators) < self.n_estimators:
            estimator = DecisionTreeClassifier()
            boot_mask = np.random.choice(len(X), size=len(X), replace=True)
            X_boot, y_boot = X[boot_mask], y[boot_mask]

            estimator.fit(X_boot, y_boot)

            oob_mask = np.full(len(X), True)
            oob_mask[boot_mask] = False
            X_oob, y_oob = X[oob_mask], y[oob_mask]

            train_score = accuracy_score(estimator.predict(X_boot), y_boot)
            test_score = accuracy_score(estimator.predict(X_oob), y_oob) if len(X_oob) > 0 else 0

            if test_score > self.threshold_test_score and train_score > self.threshold_train_score:
                self.estimators += [estimator]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], 2))
        for estimator in self.estimators:
            probas += estimator.predict_proba(X)
        return probas / len(self.estimators)