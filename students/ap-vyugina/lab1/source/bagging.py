import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


class BaggingRegressor:
    def __init__(self, n_estimators=10, base_train_threshold=0.8, base_test_threshold=0.75):
        self.n_estimators = n_estimators

        self.estimators = []
        self.data_random_seeds = []

        self.base_train_threshold = base_train_threshold
        self.base_test_threshold = base_test_threshold
    

    def fit(self, X, y):
        while len(self.estimators) < self.n_estimators:
            est = DecisionTreeRegressor()
            random_seed = np.random.randint(1, 2**16)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=random_seed)
            est.fit(X_train, y_train)

            test_score = r2_score(est.predict(X_test), y_test)
            train_score = r2_score(est.predict(X_train), y_train)

            if test_score > self.base_test_threshold and train_score > self.base_train_threshold:
                self.estimators += [est]
                self.data_random_seeds += [random_seed]

    
    def predict(self, X):
        results = []
        for estimator in self.estimators:
            results.append(estimator.predict(X))
        return np.mean(np.array(results), axis=0)