import numpy as np


class NaiveBayes:
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        _, n_features = X.shape
        
        # Априорная вероятность
        self.priors = np.zeros(self.n_classes)
        for c in range(self.n_classes):
            self.priors[c] = np.mean(y == c)
        
        # Mean и STD для каждого класса
        self.mean = np.zeros((self.n_classes, n_features))
        self.var = np.zeros((self.n_classes, n_features))
        
        for c in range(self.n_classes):
            X_c = X[y == c]
            self.mean[c, :] = np.mean(X_c, axis=0)
            self.var[c, :] = np.var(X_c, axis=0) + 1e-7
    
    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        log_priors = np.log(self.priors)
        
        for i in range(n_samples):
            log_likelihoods = np.sum(
                -0.5 * np.log(2 * np.pi * self.var)
                - 0.5 * ((X[i] - self.mean) ** 2) / self.var,
                axis=1
            )
            posteriors = log_priors + log_likelihoods
            
            y_pred[i] = np.argmax(posteriors)
            
        return y_pred

    
