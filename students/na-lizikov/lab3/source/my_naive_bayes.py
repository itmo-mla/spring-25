import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MyGaussianNB(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.means_ = None
        self.vars_ = None
        self.priors_ = None

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.means_ = np.zeros((self.n_classes_, self.n_features_))
        self.vars_ = np.zeros((self.n_classes_, self.n_features_))
        self.priors_ = np.zeros(self.n_classes_)
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx, :] = X_c.mean(axis=0)
            self.vars_[idx, :] = X_c.var(axis=0) + 1e-9
            self.priors_[idx] = X_c.shape[0] / X.shape[0]
        
        return self

    def _gaussian_likelihood(self, class_idx, x):
        mean = self.means_[class_idx]
        var = self.vars_[class_idx]
        numerator = np.exp(-0.5 * ((x - mean) ** 2) / var)
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _joint_log_likelihood(self, x):
        log_likelihoods = []
        for idx in range(self.n_classes_):
            prior = np.log(self.priors_[idx])
            class_conditional = np.sum(np.log(self._gaussian_likelihood(idx, x)))
            log_likelihoods.append(prior + class_conditional)
        return np.array(log_likelihoods)

    def predict(self, X):
        # Предсказание классов для массива X
        y_pred = []
        for x in X:
            log_likelihoods = self._joint_log_likelihood(x)
            y_pred.append(self.classes_[np.argmax(log_likelihoods)])
        return np.array(y_pred)

    def predict_proba(self, X):
        # Предсказание вероятностей классов для массива X
        proba = []
        for x in X:
            log_likelihoods = self._joint_log_likelihood(x)
            probs = np.exp(log_likelihoods - np.max(log_likelihoods))
            probs = probs / np.sum(probs)
            proba.append(probs)
        return np.array(proba)