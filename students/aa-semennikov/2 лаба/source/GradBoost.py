from sklearn.tree import DecisionTreeRegressor

class GradBoostRegressor:
    def __init__(self, loss_function, loss_grad, max_depth=3, lr=0.001, eps=0.001):
        self.estimators = []
        self.losses = []
        self.max_depth = max_depth
        self.lr = lr
        self.loss_function = loss_function
        self.loss_grad = loss_grad
        self.eps = eps
        self.num_iterations = 0

    def fit(self, X, y, n_iter=1000):
        y = y.reshape(-1, 1)
        grads = None

        for i in range(n_iter):
            estimator = DecisionTreeRegressor(max_depth=self.max_depth)

            if i == 0: # На первой итерации предсказываем таргет
                estimator.fit(X, y)
                preds = estimator.predict(X).reshape(-1, 1)
            else: # на остальных итерациях предсказываем градиент потерь, чтобы минимизировать текущую ошибку
                estimator.fit(X, grads)
                d_preds = estimator.predict(X).reshape(-1, 1)
                preds -= self.lr * d_preds

            loss = self.loss_function(y, preds) 
            grads = self.loss_grad(y, preds)
            self.losses += [loss]
            self.estimators += [estimator]
            self.num_iterations += 1

            if len(self.losses) > 3 and abs(self.losses[-2] - self.losses[-1]) < self.eps:
                break


    def predict(self, X):
        preds = self.estimators[0].predict(X).reshape(-1, 1)
        for estimator in self.estimators[1:]:
            d_preds = estimator.predict(X).reshape(-1, 1)
            preds -= self.lr * d_preds
        return preds