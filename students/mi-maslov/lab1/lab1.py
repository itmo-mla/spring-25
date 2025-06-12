import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=15)
            tree.fit(X, y, sample_weight=weights)

            y_pred = tree.predict(X)
            error = np.sum(weights[y != y_pred]) / np.sum(weights)

            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            weights *= np.exp(-alpha * y * y_pred)
            weights /= np.sum(weights)

            self.models.append(tree)
            self.alphas.append(alpha)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for alpha, tree in zip(self.alphas, self.models):
            predictions += alpha * tree.predict(X)
        return np.sign(predictions).astype(int)

boosting = AdaBoost(n_estimators=50)
start_time = time.time()
boosting.fit(X_train, y_train)
boosting_time_custom = time.time() - start_time
y_pred_boosting = boosting.predict(X_test)
print("Accuracy (Boosting):", accuracy_score(y_test, y_pred_boosting))
print(f"Custom Time (Boosting): {boosting_time_custom:.6f} seconds")

class Bagging:
    def __init__(self, n_estimators=10, max_samples=1.0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.models = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, int(self.max_samples * n_samples), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            tree = DecisionTreeClassifier()
            tree.fit(X_bootstrap, y_bootstrap)
            self.models.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.models])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

bagging = Bagging(n_estimators=2, max_samples=1.0)
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
print("Accuracy (Bagging):", accuracy_score(y_test, y_pred_bagging))

class RandomSubspaceMethod:
    def __init__(self, n_estimators=10, max_features=0.5):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.models = []
        self.feature_indices = []

    def fit(self, X, y):
        n_features = X.shape[1]
        for _ in range(self.n_estimators):
            feature_idx = np.random.choice(n_features, int(self.max_features * n_features), replace=False)
            X_subset = X[:, feature_idx]

            tree = DecisionTreeClassifier()
            tree.fit(X_subset, y)
            self.models.append(tree)
            self.feature_indices.append(feature_idx)

    def predict(self, X):
        predictions = []
        for tree, feature_idx in zip(self.models, self.feature_indices):
            predictions.append(tree.predict(X[:, feature_idx]))
        predictions = np.array(predictions)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

rsm = RandomSubspaceMethod(n_estimators=2, max_features=0.5)
rsm.fit(X_train, y_train)
y_pred_rsm = rsm.predict(X_test)
print("Accuracy (RSM):", accuracy_score(y_test, y_pred_rsm))

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier

def compare_models():
    results = {}

    # AdaBoost
    start_time = time.time()
    custom_boosting = AdaBoost(n_estimators=10)
    custom_boosting.fit(X_train, y_train)
    y_pred_custom_boosting = custom_boosting.predict(X_test)
    acc_custom_boosting = accuracy_score(y_test, y_pred_custom_boosting)
    time_custom_boosting = time.time() - start_time

    start_time = time.time()
    sklearn_boosting = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=10)
    sklearn_boosting.fit(X_train, y_train)
    y_pred_sklearn_boosting = sklearn_boosting.predict(X_test)
    acc_sklearn_boosting = accuracy_score(y_test, y_pred_sklearn_boosting)
    time_sklearn_boosting = time.time() - start_time

    results['AdaBoost'] = {
        'Custom': {'Accuracy': acc_custom_boosting, 'Time': time_custom_boosting},
        'Sklearn': {'Accuracy': acc_sklearn_boosting, 'Time': time_sklearn_boosting}
    }

    # Bagging
    start_time = time.time()
    custom_bagging = Bagging(n_estimators=10, max_samples=1.0)
    custom_bagging.fit(X_train, y_train)
    y_pred_custom_bagging = custom_bagging.predict(X_test)
    acc_custom_bagging = accuracy_score(y_test, y_pred_custom_bagging)
    time_custom_bagging = time.time() - start_time

    start_time = time.time()
    sklearn_bagging = BaggingClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=10, max_samples=1.0, bootstrap=True)
    sklearn_bagging.fit(X_train, y_train)
    y_pred_sklearn_bagging = sklearn_bagging.predict(X_test)
    acc_sklearn_bagging = accuracy_score(y_test, y_pred_sklearn_bagging)
    time_sklearn_bagging = time.time() - start_time

    results['Bagging'] = {
        'Custom': {'Accuracy': acc_custom_bagging, 'Time': time_custom_bagging},
        'Sklearn': {'Accuracy': acc_sklearn_bagging, 'Time': time_sklearn_bagging}
    }

    # Random Subspace Method
    start_time = time.time()
    custom_rsm = RandomSubspaceMethod(n_estimators=10, max_features=0.5)
    custom_rsm.fit(X_train, y_train)
    y_pred_custom_rsm = custom_rsm.predict(X_test)
    acc_custom_rsm = accuracy_score(y_test, y_pred_custom_rsm)
    time_custom_rsm = time.time() - start_time

    start_time = time.time()
    sklearn_rsm = RandomForestClassifier(n_estimators=10, max_features=0.5, bootstrap=False)
    sklearn_rsm.fit(X_train, y_train)
    y_pred_sklearn_rsm = sklearn_rsm.predict(X_test)
    acc_sklearn_rsm = accuracy_score(y_test, y_pred_sklearn_rsm)
    time_sklearn_rsm = time.time() - start_time

    results['RandomSubspaceMethod'] = {
        'Custom': {'Accuracy': acc_custom_rsm, 'Time': time_custom_rsm},
        'Sklearn': {'Accuracy': acc_sklearn_rsm, 'Time': time_sklearn_rsm}
    }

    return results

# Вывод результатов сравнения
results = compare_models()
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"  Custom Accuracy: {result['Custom']['Accuracy']:.4f}, Time: {result['Custom']['Time']:.4f} seconds")
    print(f"  Sklearn Accuracy: {result['Sklearn']['Accuracy']:.4f}, Time: {result['Sklearn']['Time']:.4f} seconds")