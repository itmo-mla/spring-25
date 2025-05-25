import numpy as np


def _gaussian_pdf(x, mean, var):
    exponent = -0.5 * ((x - mean) ** 2) / var
    return -0.5 * np.log(2 * np.pi * var) + exponent


class GaussianNaiveBayesClassifier:
    def __init__(self, epsilon=1e-9):
        self.classes = None
        self.epsilon = epsilon
        self.class_probs = {}
        self.means = {}
        self.vars = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        X.var()
        # Априорные вероятности классов
        class_counts = np.bincount(y)
        self.class_probs = class_counts / n_samples
        # Средние и дисперсии для каждого класса
        for cls in self.classes:
            X_cls = X[y == cls]
            self.means[cls] = X_cls.mean(axis=0)
            self.vars[cls] = X_cls.var(axis=0) + self.epsilon

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = np.zeros((X.shape[0], len(self.classes)))
        for i, cls in enumerate(self.classes):
            # Логарифм априорной вероятности класса
            prior = np.log(self.class_probs[cls])
            # Логарифм правдоподобия (сумма по всем признакам)
            likelihood = _gaussian_pdf(X, self.means[cls], self.vars[cls]).sum(axis=1)
            log_probs[:, i] = prior + likelihood
        return log_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self.predict_log_proba(X)
        return np.argmax(log_probs, axis=1)
