import numpy as np

class LFM:
    def __init__(self, n_users, n_items, n_factors=20, lr=0.01, reg=0.01, n_epochs=20, random_state=42):
        np.random.seed(random_state)
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = 0.0

    def fit(self, user_ids, item_ids, ratings, verbose=False):
        self.global_bias = np.mean(ratings)

        for epoch in range(self.n_epochs):
            for u, i, r in zip(user_ids, item_ids, ratings):
                pred = self.predict_single(u, i)
                err = r - pred
                # Обновление факторов
                self.user_factors[u] += self.lr * (err * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (err * self.user_factors[u] - self.reg * self.item_factors[i])
                # Обновление bias 
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])

            if verbose:
                rmse = self.rmse(user_ids, item_ids, ratings)
                print(f"Epoch {epoch+1}/{self.n_epochs}, RMSE: {rmse:.4f}")

    def predict_single(self, user_id, item_id):
        prediction = self.global_bias
        prediction += self.user_bias[user_id]
        prediction += self.item_bias[item_id]
        prediction += np.dot(self.user_factors[user_id], self.item_factors[item_id])
        return prediction

    def predict(self, user_ids, item_ids):
        return np.array([self.predict_single(u, i) for u, i in zip(user_ids, item_ids)])

    def rmse(self, user_ids, item_ids, ratings):
        preds = self.predict(user_ids, item_ids)
        return np.sqrt(np.mean((ratings - preds) ** 2))

    def mae(self, user_ids, item_ids, ratings):
        preds = self.predict(user_ids, item_ids)
        return np.mean(np.abs(ratings - preds)) 