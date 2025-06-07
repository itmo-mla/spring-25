import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class LFM:
    def __init__(self, n_users, n_items, n_factors=10, lr=0.01, reg_coef=0.02, n_epochs=20):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg_coef = reg_coef
        self.n_epochs = n_epochs
        # Инициализируем матрицы латентных факторов
        self.user_factors = np.random.normal(scale=1 / n_factors, size=(n_users, n_factors))
        self.item_factors = np.random.normal(scale=1 / n_factors, size=(n_items, n_factors))

    def fit(self, data):
        for _ in range(self.n_epochs):
            for _, row in data.iterrows():
                u = row['user_idx']
                i = row['item_idx']
                r_ui = row['rating']
                # Считаем предикт как произведение вектора л. факторов пользователя на вектор л. факторов фильма
                pred = self.user_factors[u].dot(self.item_factors[i].T)
                # Разница между реальным рейтингом и предиктом
                error = r_ui - pred
                # u_new = u + lr * (error * i - reg_coef * u)
                self.user_factors[u] += self.lr * (error * self.item_factors[i] - self.reg_coef * self.user_factors[u])
                # i_new = i + lr * (error * u - reg_coef * i)
                self.item_factors[i] += self.lr * (error * self.user_factors[u] - self.reg_coef * self.item_factors[i])

    def predict(self, user_idx, item_idx):
        return self.user_factors[user_idx].dot(self.item_factors[item_idx].T)

    def evaluate(self, data):
        preds, y_true = [], []
        for _, row in data.iterrows():
            u = row['user_idx']
            i = row['item_idx']
            r_ui = row['rating']
            y_true.append(r_ui)
            pred = self.predict(u, i)
            preds.append(pred)
        rmse = np.sqrt(mean_squared_error(preds, y_true))
        mae = mean_absolute_error(preds, y_true)
        return rmse, mae