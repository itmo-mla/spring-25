import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(self, loss_func, loss_grad, max_depth=3, lr=1e-2, eps=1e-3):
        self.estimators = []
        self.losses = []

        self.max_depth = max_depth
        self.lr = lr

        self.loss_func = loss_func
        self.loss_grad = loss_grad

        self.eps = eps
        self.num_iterations = 0

    def fit(self, X, y, n_iter=1000):
        y = y.reshape(-1, 1)
        grads = None

        for _ in range(n_iter):

            est = DecisionTreeRegressor(max_depth=self.max_depth)
            if grads is None: # the first iteration predicts y_pred values
                est.fit(X, y)
                preds = est.predict(X).reshape(-1, 1)
            else: # any other iteration predicts d_y_pred
                est.fit(X, grads)

                d_preds = est.predict(X).reshape(-1, 1)
                preds -= self.lr * d_preds

            loss = self.loss_func(y, preds) 
            grads = self.loss_grad(y, preds)

            self.losses += [loss]
            self.estimators += [est]
            self.num_iterations += 1

            # stopping criteria
            if len(self.losses) > 3 and abs(self.losses[-2] - self.losses[-1]) < self.eps:
                break

    
    def predict(self, X):
        preds = self.estimators[0].predict(X).reshape(-1, 1)
        for estimator in self.estimators[1:]:
            d_preds = estimator.predict(X).reshape(-1, 1)
            preds -= self.lr * d_preds
        return preds
    

class StochasticGradientBoostingRegressor(GradientBoostingRegressor):
    def fit(self, X, y, n_iter=1000):
        y = y.reshape(-1, 1)
        grads = None
        idxs = np.arange(len(X))

        for _ in range(n_iter):
            random_seed = np.random.randint(1, 2**16)
            train_idxs, _ = train_test_split(idxs, test_size=1/3, random_state=random_seed)

            est = DecisionTreeRegressor(max_depth=self.max_depth)
            if grads is None: # the first iteration predicts y_pred values
                est.fit(X[train_idxs], y[train_idxs])
                preds = est.predict(X).reshape(-1, 1)

            else: # any other iteration predicts d_y_pred
                est.fit(X[train_idxs], grads[train_idxs])

                d_preds = est.predict(X).reshape(-1, 1)
                preds -= self.lr * d_preds

            loss = self.loss_func(y, preds) 
            grads = self.loss_grad(y, preds)
            self.num_iterations += 1

            self.losses += [loss]
            self.estimators += [est]

            if len(self.losses) > 3 and abs(self.losses[-2] - self.losses[-1]) < self.eps:
                break