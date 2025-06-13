import numpy as np

class NaiveBayesClassifier:
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        n_features = X.shape[1]
        self.priors = np.zeros(self.n_classes)
        self.mean = np.zeros((self.n_classes, n_features))
        self.var = np.zeros((self.n_classes, n_features))

        for klass in range(self.n_classes):
            # Считаем априорную вероятность для каждого класса
            self.priors[klass] = np.mean(y == klass)
            # Считаем среднее и стандартное отклонение по всем признакам для каждого класса
            self.mean[klass, :] = np.mean(X[y == klass], axis=0)
            self.var[klass, :] = np.var(X[y == klass], axis=0)

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        # Логарифмируем априорные вероятности 
        log_priors = np.log(self.priors)
        
        for i in range(n_samples):
            # Для каждого примера считаем логарифм правдоподобия принадлежности к каждому классу
            log_likelihoods = np.sum(-0.5*np.log(2 * np.pi * self.var) - 0.5*((X[i] - self.mean) ** 2) / self.var, axis=1)
            posteriors = log_priors + log_likelihoods
            y_pred[i] = np.argmax(posteriors)
            
        return y_pred