import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import time

df = pd.read_csv('./datasets/SVMtrain.csv')
df = df.drop(columns=['PassengerId', 'Embarked'], axis=1)
df['Sex'] = df['Sex'].map({'female': 0, 'Male': 1})

X = df.drop(['Survived'], axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)


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

class EpanechnikovNB:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.X_by_class = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.X_by_class[c] = X_c
            self.priors[c] = len(X_c) / len(X)

    def _epanechnikov_kernel(self, u):
        mask = np.abs(u) <= 1
        return 0.75 * (1 - u**2) * mask

    def _calculate_likelihood(self, class_idx, x):
        X_c = self.X_by_class[class_idx]
        h = self.bandwidth
        n = len(X_c)

        log_density = 0.0
        for x_i, X_i in zip(x, X_c.T.values):
            u = (x_i - X_i) / h
            k_vals = self._epanechnikov_kernel(u)
            density = np.sum(k_vals) / (n * h)
            log_density += np.log(density + 1e-10)

        return log_density

    def _calculate_posterior(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = self._calculate_likelihood(c, x)
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._calculate_posterior(x) for x in X.values])


class CustomGaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-9
            self.priors[c] = len(X_c) / len(X)

    def _calculate_likelihood(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]

        log_likelihood = 0.0
        for x_i, mean_i, var_i in zip(x, mean.values, var.values):
            likelihood_i = (1 / np.sqrt(2 * np.pi * var_i)
                            * np.exp(-((x_i - mean_i) ** 2) / (2 * var_i)))
            log_likelihood += np.log(likelihood_i + 1e-10)

        return log_likelihood

    def _calculate_posterior(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = self._calculate_likelihood(c, x)
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._calculate_posterior(x) for x in X.values])

bandwidths = np.linspace(0.1, 2.0, 20)
best_score = -np.inf
best_bandwidth = None

print("\nSearching for best bandwidth:")
for bw in bandwidths:
    score = custom_cv_score(EpanechnikovNB, X, y, cv=5, bandwidth=bw)
    print(f"Bandwidth = {bw:.2f}, CV Accuracy = {score:.4f}")
    if score > best_score:
        best_score = score
        best_bandwidth = bw

print(f"\nBest bandwidth: {best_bandwidth:.2f} with CV Accuracy = {best_score:.4f}")

start_time = time.time()
epan_nb = EpanechnikovNB(bandwidth=best_bandwidth)
epan_nb.fit(X_train, y_train)
epan_training_time = time.time() - start_time

epan_predictions = epan_nb.predict(X_test)
epan_accuracy = accuracy_score(y_test, epan_predictions)

epan_cv_mean = custom_cv_score(EpanechnikovNB, X, y, cv=5, bandwidth=best_bandwidth)

print(f"\nEPANECHNIKOV")
print(f'EpanechnikovNB Accuracy: {epan_accuracy:.4f}')
print(f'EpanechnikovNB Cross-Validation Accuracy: {epan_cv_mean:.4f}')
print(f'EpanechnikovNB Training Time: {epan_training_time:.4f} seconds')


start_time = time.time()
custom_nb = CustomGaussianNB()
custom_nb.fit(X_train, y_train)
custom_training_time = time.time() - start_time

custom_predictions = custom_nb.predict(X_test)
custom_accuracy = accuracy_score(y_test, custom_predictions)

custom_cv_mean = custom_cv_score(CustomGaussianNB, X, y, cv=5)



start_time = time.time()
sklearn_nb = GaussianNB()
sklearn_nb.fit(X_train, y_train)
sklearn_training_time = time.time() - start_time

sklearn_predictions = sklearn_nb.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
sklearn_cv_scores = cross_val_score(sklearn_nb, X, y, cv=5)
sklearn_cv_mean = sklearn_cv_scores.mean()

print(f"\nCUSTOM")
print(f'Custom GaussianNB Accuracy: {custom_accuracy:.4f}')
print(f'Custom GaussianNB Cross-Validation Accuracy: {custom_cv_mean:.4f}')
print(f'Custom GaussianNB Training Time: {custom_training_time:.4f} seconds')

print(f"\nSKLEARN")
print(f'Sklearn GaussianNB Accuracy: {sklearn_accuracy:.4f}')
print(f'Sklearn GaussianNB Cross-Validation Accuracy: {sklearn_cv_mean:.4f}')
print(f'Sklearn GaussianNB Training Time: {sklearn_training_time:.4f} seconds')
