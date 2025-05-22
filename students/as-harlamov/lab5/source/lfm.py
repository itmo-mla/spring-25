import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class LatentFactorModel:
    def __init__(self, n_users, n_items, n_factors=10, lr=0.01, reg=0.02, n_epochs=20):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs

        self.user_factors = np.random.normal(scale=1. / n_factors, size=(n_users, n_factors))
        self.item_factors = np.random.normal(scale=1. / n_factors, size=(n_items, n_factors))

    def fit(self, train_data):
        for epoch in range(self.n_epochs):
            for _, row in train_data.iterrows():
                u = row['user_idx']
                i = row['item_idx']
                r_ui = row['rating']

                pred = self.user_factors[u].dot(self.item_factors[i].T)

                err = r_ui - pred

                self.user_factors[u] += self.lr * (err * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (err * self.user_factors[u] - self.reg * self.item_factors[i])

    def predict(self, user_idx, item_idx):
        return self.user_factors[user_idx].dot(self.item_factors[item_idx].T)

    def evaluate(self, test_data):
        preds = []
        true = []
        for _, row in test_data.iterrows():
            u = row['user_idx']
            i = row['item_idx']
            r_ui = row['rating']
            pred = self.predict(u, i)
            preds.append(pred)
            true.append(r_ui)
        rmse = np.sqrt(mean_squared_error(true, preds))
        mae = mean_absolute_error(true, preds)
        return rmse, mae
