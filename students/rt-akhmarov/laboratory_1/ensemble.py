import numpy as np

from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier(BaseEnsemble):
    def __init__(
            self, 
            estimator:DecisionTreeClassifier = None, 
            n_estimators = 10, 
            random_state = 42, 
            *, 
            estimator_params = ...):
        super().__init__(estimator, n_estimators=n_estimators, estimator_params=estimator_params)
        self.estimator_ = estimator
        self.n_estimators_ = n_estimators
        self.estimators_ = []
        self.classes_ = []
        self.selected_features_list_ = []
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    def fit(self, X:np.ndarray, y:np.ndarray):
        self.classes_ = np.unique(y)

        for _ in range(self.n_estimators_):    
            X_train, y_train = self._bootstrap_sample(X, y)
            # X_train, feature_indices = self._subspace_sample(X_train, return_indices=True)
            # self.selected_features_list_.append(feature_indices)

            tree = self.estimator_(max_depth=None, max_features='sqrt')
            tree.fit(X_train, y_train)
            self.estimators_.append(tree)
    
    def predict(self, X: np.ndarray):
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int), minlength=len(self.classes_)).argmax(), 
            axis=0, 
            arr=predictions
            )

    # def predict(self, X:np.ndarray):
    #     predictions = np.zeros((X.shape[0], self.n_estimators_))
    #     for j in range(self.n_estimators_):
    #         X_subset = X[:, self.selected_features_list_[j]]
    #         predictions[:,j] = self.estimators_[j].predict(X_subset)

    #     return np.array(
    #         [np.sum(predictions == cls, axis=-1) for cls in self.classes_]
    #     ).argmax(axis=0).T
    
    def _bootstrap_sample(self, X:np.ndarray, y:np.ndarray):
        n_samples = X.shape[0]
        indices = self.rng.integers(0, n_samples, size=n_samples)
        return X[indices], y[indices]

    # def _subspace_sample(self, X:np.ndarray, return_indices=True):
    #     n_features = X.shape[1]
    #     n_subspace = int(np.sqrt(n_features))
    #     feature_indices = self.rng.choice(n_features, size=n_subspace, replace=False)
    #     self.selected_features_ = feature_indices  
    #     if return_indices:
    #         return X[:, feature_indices], feature_indices
    #     else:
    #         return X[:, feature_indices]

