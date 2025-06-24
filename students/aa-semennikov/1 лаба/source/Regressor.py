import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


class BaggingRegressor:
    def __init__(self, n_estimators=10, train_threshold=0.85, test_threshold=0.8):
        self.n_estimators = n_estimators
        self.train_threshold = train_threshold
        self.test_threshold = test_threshold
        self.estimators = []   

    def fit(self, X, y):
        while len(self.estimators) < self.n_estimators:
            estimator = DecisionTreeRegressor()
            seed = np.random.randint(1, 1024)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=seed)
            estimator.fit(X_train, y_train)
            test_score = r2_score(estimator.predict(X_test), y_test)
            train_score = r2_score(estimator.predict(X_train), y_train)

            if test_score > self.test_threshold and train_score > self.train_threshold:
                self.estimators += [estimator]

    def predict(self, X):
        results = []
        for estimator in self.estimators:
            results.append(estimator.predict(X))
        return np.mean(np.array(results), axis=0)