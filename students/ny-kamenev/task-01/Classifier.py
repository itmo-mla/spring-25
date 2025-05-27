import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import time
from joblib import Parallel, delayed

class CustomBaggingClassifier:
    def __init__(self, n_estimators=100, max_depth=5, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.models = []

    def _train_single_model(self, X_sample, y_sample, X_oob, y_oob):
        model = DecisionTreeClassifier(max_depth=self.max_depth)
        model.fit(X_sample, y_sample)

        if len(y_oob) > 0:
            oob_preds = model.predict(X_oob)
            if accuracy_score(y_oob, oob_preds) > 0.5:
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
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

df = pd.read_csv('./datasets/SVMtrain.csv')
df = df.drop(columns=['PassengerId', 'Embarked'], axis=1)
df['Sex'] = df['Sex'].map({'female': 0, 'Male': 1})

X = df.drop(['Survived'], axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

start_time = time.time()
custom_bagging = CustomBaggingClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
custom_bagging.fit(X_train, y_train)
end_time = time.time()
custom_training_time = end_time - start_time

custom_predictions = custom_bagging.predict(X_test)
custom_accuracy = accuracy_score(y_test, custom_predictions)

def custom_cv_score(model, X, y, cv=5):
    scores = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=33)
    for train_idx, test_idx in kf.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        scores.append(accuracy_score(y.iloc[test_idx], preds))
    return np.mean(scores)

custom_cv_mean = custom_cv_score(custom_bagging, X, y, cv=5)

start_time = time.time()
sklearn_bagging = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=10), n_estimators=100, random_state=33)
sklearn_bagging.fit(X_train, y_train)
end_time = time.time()
sklearn_training_time = end_time - start_time

sklearn_predictions = sklearn_bagging.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
sklearn_cv_scores = cross_val_score(sklearn_bagging, X, y, cv=5)
sklearn_cv_mean = sklearn_cv_scores.mean()


print(f"\nCUSTOM")
print(f'Custom Bagging Accuracy: {custom_accuracy:.4f}')
print(f'Custom Bagging Cross-Validation Accuracy: {custom_cv_mean:.4f}')
print(f'Custom Bagging Training Time: {custom_training_time:.4f} seconds')
print(f"\nSKLEARN")
print(f'Sklearn Bagging Accuracy: {sklearn_accuracy:.4f}')
print(f'Sklearn Bagging Cross-Validation Accuracy: {sklearn_cv_mean:.4f}')
print(f'Sklearn Bagging Training Time: {sklearn_training_time:.4f} seconds')
