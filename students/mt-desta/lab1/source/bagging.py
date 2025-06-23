import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


class BaggingRegressor:

    def __init__(self, n_estimators=10, max_depth = 10, threshold = 0.5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.threshold = threshold


    def fit(self, X, y):
        self.models = []
        for _ in range(self.n_estimators):
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(0,10000))
            
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            if r2 > self.threshold:
                self.models.append(model)


    def predict(self, X):
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return predictions / len(self.models)





