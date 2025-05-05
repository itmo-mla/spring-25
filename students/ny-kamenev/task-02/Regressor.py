import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import time

DEPTH = 5

df = pd.read_csv('./datasets/SVMtrain.csv')
df = df.drop(columns=['PassengerId', 'Embarked'], axis=1)
df['Sex'] = df['Sex'].map({'female': 0, 'Male': 1})

X = df.drop(['Survived'], axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

class CustomGradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.f0 = None

    def fit(self, X, y):
        self.f0 = np.mean(y)
        residuals = y - self.f0

        for _ in range(self.n_estimators):
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, residuals)
            predictions = model.predict(X)
            residuals -= self.learning_rate * predictions
            self.models.append(model)

    def predict(self, X):
        predictions = np.full(X.shape[0], self.f0)
        for model in self.models:
            predictions += self.learning_rate * model.predict(X)
        return np.round(predictions)


start_time = time.time()
custom_gb = CustomGradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=DEPTH)
custom_gb.fit(X_train, y_train)
custom_training_time = time.time() - start_time

custom_predictions = custom_gb.predict(X_test)
custom_accuracy = accuracy_score(y_test, custom_predictions)

def custom_cv_score(model_class, X, y, cv=5, **model_params):
    scores = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=33)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        scores.append(accuracy_score(y_test, preds))

    return np.mean(scores)

custom_cv_mean = custom_cv_score(CustomGradientBoosting, X, y, cv=5, n_estimators=100, learning_rate=0.1, max_depth=DEPTH)

start_time = time.time()
sklearn_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=DEPTH, random_state=33)
sklearn_gb.fit(X_train, y_train)
sklearn_training_time = time.time() - start_time

sklearn_predictions = sklearn_gb.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
sklearn_cv_scores = cross_val_score(sklearn_gb, X, y, cv=5)
sklearn_cv_mean = sklearn_cv_scores.mean()

print(f"\nCUSTOM")
print(f'Custom Gradient Boosting Accuracy: {custom_accuracy:.4f}')
print(f'Custom Gradient Boosting Cross-Validation Accuracy: {custom_cv_mean:.4f}')
print(f'Custom Gradient Boosting Training Time: {custom_training_time:.4f} seconds')
print(f"\nSKLEARN")
print(f'Sklearn Gradient Boosting Accuracy: {sklearn_accuracy:.4f}')
print(f'Sklearn Gradient Boosting Cross-Validation Accuracy: {sklearn_cv_mean:.4f}')
print(f'Sklearn Gradient Boosting Training Time: {sklearn_training_time:.4f} seconds')
