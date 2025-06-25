import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from tqdm import trange
import matplotlib.pyplot as plt

class Processor:
    def __init__(self, rating_df, anime_df):
        self.rating_df = rating_df
        self.anime_df = anime_df
        self.id_to_name = self._build_mapper()
        self.user_item_matrix = self._build_sparse_matrix()

    def _build_mapper(self):
        return dict(zip(self.anime_df['anime_id'], self.anime_df['English name']))

    def _build_sparse_matrix(self):
        users = self.rating_df['user_id'].unique()
        items = self.rating_df['anime_id'].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {j: k for k, j in enumerate(items)}
        n_users = len(users)
        n_items = len(items)
        rows = self.rating_df['user_id'].map(self.user_index)
        cols = self.rating_df['anime_id'].map(self.item_index)
        data = self.rating_df['rating'].values
        return sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

    def split(self, test_size=0.2, random_state=None):
        coo = self.user_item_matrix.tocoo()
        idx = np.arange(len(coo.data))
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
        train_mat = sp.csr_matrix(
            (coo.data[train_idx], (coo.row[train_idx], coo.col[train_idx])), shape=coo.shape
        )
        test_mat = sp.csr_matrix(
            (coo.data[test_idx], (coo.row[test_idx], coo.col[test_idx])), shape=coo.shape
        )
        return train_mat, test_mat

    def id_to_name_list(self, anime_ids):
        return [self.id_to_name.get(i) for i in anime_ids]

class LatentFactorModel:
    def __init__(self, n_factors: int = 20, n_epochs: int = 10, lr: float = 0.01,
                 reg: float = 0.1, reg_b: float = 0.1):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg         
        self.reg_b = reg_b     

        self.global_bias = None
        self.user_bias = None
        self.item_bias = None
        self.user_factors = None
        self.item_factors = None
        self.train_matrix = None

        self.train_rmse = []
        self.train_mae = []
        self.train_loss = []

    def fit(self, X):
        self.train_matrix = X
        n_users, n_items = X.shape

        coo = X.tocoo()
        self.global_bias = np.mean(coo.data)

        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.user_factors = np.random.normal(scale=1. / self.n_factors,
                                             size=(n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=1. / self.n_factors,
                                             size=(n_items, self.n_factors))

        rows, cols = coo.row, coo.col
        pbar = trange(self.n_epochs, desc='Epoch')

        for epoch in pbar:
            mse_sum = 0.0
            mae_sum = 0.0
            for u, i in zip(rows, cols):
                pred = (self.global_bias + self.user_bias[u] + self.item_bias[i] +
                        self.user_factors[u].dot(self.item_factors[i]))
                err = X[u, i] - pred

                self.user_bias[u] += self.lr * (err - self.reg_b * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg_b * self.item_bias[i])

                self.user_factors[u] += self.lr * (
                    err * self.item_factors[i] - self.reg * self.user_factors[u]
                )
                self.item_factors[i] += self.lr * (
                    err * self.user_factors[u] - self.reg * self.item_factors[i]
                )

                mse_sum += err ** 2
                mae_sum += abs(err)

            n_samples = len(rows)
            mse = mse_sum / n_samples
            mae = mae_sum / n_samples

            reg_term = (self.reg * (np.linalg.norm(self.user_factors) ** 2 +
                                     np.linalg.norm(self.item_factors) ** 2) +
                        self.reg_b * (np.linalg.norm(self.user_bias) ** 2 +
                                      np.linalg.norm(self.item_bias) ** 2))
            loss = mse + reg_term / (n_users + n_items)

            self.train_rmse.append(np.sqrt(mse))
            self.train_mae.append(mae)
            self.train_loss.append(loss)
            pbar.set_postfix({'rmse': np.sqrt(mse), 'mae': mae, 'loss': loss})

        return self

    def predict(self, user_index, item_index):
        return (self.global_bias + self.user_bias[user_index] + self.item_bias[item_index] +
                self.user_factors[user_index].dot(self.item_factors[item_index]))

    def predict_topK(self, user_index, K=10):
        scores = (self.global_bias + self.user_bias[user_index] + self.item_bias +
                  self.user_factors[user_index].dot(self.item_factors.T))
        seen = set(self.train_matrix[user_index].nonzero()[1])
        candidates = [(i, s) for i, s in enumerate(scores) if i not in seen]
        topK = sorted(candidates, key=lambda x: x[1], reverse=True)[:K]
        items, scores = zip(*topK) if topK else ([], [])
        return list(items), list(scores)

    def plot_loss(self):
        epochs = range(1, len(self.train_loss) + 1)
        plt.figure()
        plt.plot(epochs, self.train_loss, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.show()

    def plot_metrics(self):
        epochs = range(1, len(self.train_loss) + 1)
        plt.figure()
        plt.plot(epochs, self.train_rmse, label='RMSE')
        plt.plot(epochs, self.train_mae, label='MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Training RMSE & MAE')
        plt.legend()
        plt.show()

    def evaluate(self, test_matrix):
        coo = test_matrix.tocoo()
        preds = (
            self.global_bias + self.user_bias[coo.row] + self.item_bias[coo.col] +
            np.sum(self.user_factors[coo.row] * self.item_factors[coo.col], axis=1)
        )
        errs = coo.data - preds
        rmse = np.sqrt(np.mean(errs ** 2))
        mae = np.mean(np.abs(errs))
        return {'rmse': rmse, 'mae': mae}
