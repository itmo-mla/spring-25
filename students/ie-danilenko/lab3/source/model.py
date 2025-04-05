import numpy as np

# distribution = gaussian or multinomial
class NaiveBayes:
    def __init__(self, distribution='gaussian'):
        self.distribution = distribution
        self.classes = None
        self.class_priors = None
        self.feature_params = None
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        self.class_priors = {}
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
            
        self.feature_params = {}
        for c in self.classes:
            X_c = X[y == c]
            
            if self.distribution == 'gaussian':
                self.feature_params[c] = {
                    'mean': np.mean(X_c, axis=0),
                    'var': np.var(X_c, axis=0) + 1e-9 
                }
            else:
                self.feature_params[c] = {
                    'counts': np.sum(X_c, axis=0),
                    'total': np.sum(X_c)
                }
                
    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])
    
    def _predict_single(self, x):
        posteriors = {}
        
        for c in self.classes:
            posterior = np.log(self.class_priors[c])
            
            if self.distribution == 'gaussian':
                mean = self.feature_params[c]['mean']
                var = self.feature_params[c]['var']
                
                posterior += np.sum(-0.5 * np.log(2 * np.pi * var) - 
                                  0.5 * ((x - mean) ** 2) / var)
            else:  # multinomial
                counts = self.feature_params[c]['counts']
                total = self.feature_params[c]['total']
                posterior += np.sum(x * np.log((counts + 1) / (total + len(x))))
                
            posteriors[c] = posterior
            
        return max(posteriors.items(), key=lambda x: x[1])[0]

class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def __split(self, X):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)
        
        fold_sizes = (n_samples // self.n_splits) * np.ones(self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, end = current, current + fold_size
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            yield train_indices, val_indices
            current = end

    def cros_valid(self, model, X, y, metrics):
        metric_scores = []

        for train_idx, val_idx in self.__split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            metric_scores.append(metrics(y_pred, y_val))
        
        return np.asarray(metric_scores)

if __name__ == "__main__":
    from read import read_heart
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from time import time

    X, y = read_heart('data/heart.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = NaiveBayes()
    fold = KFold(n_splits=5, shuffle=True, random_state=42)
    start_time = time()
    scores = fold.cros_valid(model, X_train, y_train, mean_squared_error)
    print(f"Train time: {time() - start_time}")
    print("Cross-validation MSE:", np.mean(scores))
