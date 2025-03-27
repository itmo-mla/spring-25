import numpy as np

from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        current_prediction = np.full(y.shape, fill_value=self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = y - current_prediction

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            predictions_update = tree.predict(X)
            current_prediction += self.learning_rate * predictions_update

    def predict(self, X):
        prediction = np.full(X.shape[0], fill_value=self.initial_prediction)
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)
        return prediction
