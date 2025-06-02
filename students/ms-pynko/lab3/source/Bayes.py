import numpy as np


class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
        self.n_classes = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        self.n_classes = len(self.classes)
        
        self.mean = np.zeros((self.n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((self.n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(self.n_classes, dtype=np.float64)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = np.mean(X_c, axis=0)
            self.var[idx, :] = np.var(X_c, axis=0) + 1e-9
            self.priors[idx] = X_c.shape[0] / float(n_samples)
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        for idx in range(self.n_classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._true(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def _true(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        p_x_y = np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
        return p_x_y + 1e-9
