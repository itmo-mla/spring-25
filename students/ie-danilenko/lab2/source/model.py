import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        self.__random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        current_predictions = np.full(y.shape, self.initial_prediction)
        
        for _ in range(self.n_estimators):
            residuals = y - current_predictions
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.__random_state)
            tree.fit(X, residuals)
            current_predictions += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X)
        pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred
    
class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def __split(self, X):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
        
        fold_sizes = (n_samples // self.n_splits) * np.ones(self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, end = current, current + fold_size
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            yield train_indices, val_indices
            current = end

    def cros_valid(self, model, X, y, metrics):
        metric_scores = []

        for train_idx, val_idx in self.__split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            metric_scores.append(metrics(y_pred, y_val))
        
        return np.asarray(metric_scores)


if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from read import read_data

    X, y = read_data("data/possum.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=16, learning_rate=0.005)
    model.fit(X_train, y_train)

    fold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = fold.cros_valid(model, X_test, y_test, mean_squared_error)
    print("Cross-validation MSE:", np.mean(scores))