import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import time

data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class CustomGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.init_val = None

    def fit(self, X, y):
        self.models = []
        self.init_val = np.mean(y)
        y_pred = np.full(y.shape, self.init_val)

        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.models.append(tree)
            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.init_val)
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

start_time = time.time()
custom_model = CustomGradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
custom_model.fit(X_train, y_train)
custom_train_time = time.time() - start_time

y_pred_custom = custom_model.predict(X_test)
custom_mse = mean_squared_error(y_test, y_pred_custom)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
custom_cv_mse = []
custom_cv_times = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = CustomGradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    start_cv = time.time()
    model.fit(X_tr, y_tr)
    duration_cv = time.time() - start_cv
    y_val_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)

    custom_cv_mse.append(mse)
    custom_cv_times.append(duration_cv)

custom_cv_mse_mean = np.mean(custom_cv_mse)
custom_cv_time_mean = np.mean(custom_cv_times)

start_time = time.time()
sklearn_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
sklearn_model.fit(X_train, y_train)
sklearn_train_time = time.time() - start_time

y_pred_sklearn = sklearn_model.predict(X_test)
sklearn_mse = mean_squared_error(y_test, y_pred_sklearn)

sk_cv_scores = cross_val_score(sklearn_model, X, y, scoring='neg_mean_squared_error', cv=5)
sk_cv_mse_mean = -np.mean(sk_cv_scores)

print("\nКастомная реализация")
print(f"MSE на тестовой выборке: {custom_mse:.4f}")
print(f"Средняя MSE по кросс-валидации: {custom_cv_mse_mean:.4f}")
print(f"Среднее время обучения: {custom_train_time:.6f} сек")

print("\nРеализация scikit-learn")
print(f"MSE на тестовой выборке: {sklearn_mse:.4f}")
print(f"Средняя MSE по кросс-валидации: {sk_cv_mse_mean:.4f}")
print(f"Среднее время обучения: {sklearn_train_time:.6f} сек")

