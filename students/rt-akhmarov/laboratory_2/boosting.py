import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator

class GradientBoosting(BaseEstimator):
    def __init__(
        self, 
        n_estimators:int = 10,
        max_depth:int = 3,
        learning_rate:float = .1,
        random_state:int = 42,
        ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._estimators = []
        self._losses = []

    def _one_hot(self, y):
        _classes = np.unique(y)
        _n_classes = len(_classes)

        cls_to_idx = {cls:idx for idx, cls in enumerate(_classes)}
        y_idx = np.array([cls_to_idx[cls] for cls in y])

        y_oh = np.eye(N=len(y), M=_n_classes)[y_idx]
        return y_oh

    def _softmax(self, scores):
        exp_scores = np.exp(scores)
        sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
        return exp_scores / sum_exp
    
    def _cross_entropy_loss(self, y_true, logits):
        p = self._softmax(logits)
        log_p = np.log(p + 1e-15)
        loss = -np.mean(np.sum(y_true * log_p, axis=1))
        return loss
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        if y.ndim == 1:
            y = self._one_hot(y)

        self._classes = np.unique(y)
        self._n_classes = y.shape[1]

        F = np.zeros_like(y, dtype=np.float32)

        loss_value = self._cross_entropy_loss(y, F)
        self._losses.append(loss_value)
        
        for _ in range(self.n_estimators):
            # P = self._softmax(F)
            r = y - F

            estimator = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            estimator.fit(X, r)

            F += self.learning_rate * estimator.predict(X)

            self._estimators.append(estimator)

            loss_value = self._cross_entropy_loss(y, F)
            self._losses.append(loss_value)

    def predict_proba(self, X):
        F_final = np.zeros(shape=(X.shape[0], self._n_classes), dtype=np.float64)

        for estimator in self._estimators:
            F_final += self.learning_rate * estimator.predict(X)



        return self._softmax(F_final)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        _idx = np.argmax(probs, axis=1)
        return _idx
