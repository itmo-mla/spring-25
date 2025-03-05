import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import time
from joblib import Parallel, delayed

class CustomBaggingRegressor:
    def __init__(self, n_estimators=100, max_depth=5, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.models = []

    def _train_single_model(self, X_train, y_train, X_val, y_val):
        model = DecisionTreeRegressor(max_depth=self.max_depth)
        model.fit(X_train, y_train)

        if len(y_val) > 0:
            val_preds = model.predict(X_val)
            if mean_squared_error(y_val, val_preds) < np.var(y_val):
                return model
        return None

    def fit(self, X, y):
        n_samples = int(len(X) * 0.632)
        data_indices = np.arange(len(X))

        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(self._train_single_model)(
                X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx]
            )
            for _ in range(self.n_estimators)
            for train_idx in [np.random.choice(data_indices, n_samples, replace=True)]
            for val_idx in [np.setdiff1d(data_indices, train_idx, assume_unique=True)]
        )

        self.models = [model for model in results if model is not None]

    def predict(self, X):
        if not self.models:
            raise ValueError("Нет моделей в ансамбле")
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

df = pd.read_csv('./datasets/boston.csv')
X = df.drop(['MEDV'], axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)


start_time = time.time()
custom_bagging = CustomBaggingRegressor(n_estimators=100, max_depth=7, n_jobs=-1)
custom_bagging.fit(X_train, y_train)
end_time = time.time()
custom_training_time = end_time - start_time

custom_predictions = custom_bagging.predict(X_test)
custom_mse = mean_squared_error(y_test, custom_predictions)

def custom_cv_score(model, X, y, cv=5):
    scores = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=33)
    for train_idx, test_idx in kf.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        scores.append(mean_squared_error(y.iloc[test_idx], preds))
    return np.mean(scores)

custom_cv_mean = custom_cv_score(custom_bagging, X, y, cv=5)

start_time = time.time()
sklearn_bagging = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=7), n_estimators=100, random_state=33)
sklearn_bagging.fit(X_train, y_train)
end_time = time.time()
sklearn_training_time = end_time - start_time

sklearn_predictions = sklearn_bagging.predict(X_test)
sklearn_mse = mean_squared_error(y_test, sklearn_predictions)


sklearn_cv_scores = cross_val_score(sklearn_bagging, X, y, cv=5, scoring='neg_mean_squared_error')
sklearn_cv_mean = -sklearn_cv_scores.mean()

print(f"\nCUSTOM")
print(f'Custom Bagging MSE: {custom_mse:.4f}')
print(f'Custom Bagging Cross-Validation MSE: {custom_cv_mean:.4f}')
print(f'Custom Bagging Training Time: {custom_training_time:.4f} seconds')

print(f"\nSKLEARN")
print(f'Sklearn Bagging MSE: {sklearn_mse:.4f}')
print(f'Sklearn Bagging Cross-Validation MSE: {sklearn_cv_mean:.4f}')
print(f'Sklearn Bagging Training Time: {sklearn_training_time:.4f} seconds')
