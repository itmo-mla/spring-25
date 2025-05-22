import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SVDCustom:
    """Реализация алгоритма SVD """

    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02, reg_biases=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.reg_biases = reg_biases

    def fit(self, X_train):
        self.n_users = np.max(X_train[:, 0]) + 1
        self.n_items = np.max(X_train[:, 1]) + 1

        # Инициализация P и Q
        self.P = np.random.normal(scale=1. / self.n_factors, size=(self.n_users, self.n_factors))  # Users
        self.Q = np.random.normal(scale=1. / self.n_factors, size=(self.n_items, self.n_factors))  # Items

        self.global_avg = np.mean(X_train[:, 2])

        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)

        for _ in range(self.n_epochs):
            np.random.shuffle(X_train)
        for u, i, r in X_train:
            pred = self.global_avg + self.user_bias[u] + self.item_bias[i] + np.dot(self.P[u], self.Q[i])
            error = r - pred

            self.user_bias[u] += self.lr * (error - self.reg_biases * self.user_bias[u])
            self.item_bias[i] += self.lr * (error - self.reg_biases * self.item_bias[i])

            self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
            self.Q[i] += self.lr * (error * self.P[u] - self.reg * self.Q[i])

    def predict(self, u, i):
        return self.global_avg + self.user_bias[u] + self.item_bias[i] + np.dot(self.P[u], self.Q[i])

    def test(self, test_data):
        """Оценка качества на test_data (список кортежей (user_id, item_id, rating))"""
        predictions = [self.predict(u, i) for u, i, _ in test_data]
        true_ratings = [r for _, _, r in test_data]

        rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
        mae = mean_absolute_error(true_ratings, predictions)

        return rmse, mae
