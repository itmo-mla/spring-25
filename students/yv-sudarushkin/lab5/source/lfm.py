import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error #, mean_absolute_error


class LatentFactorModel:
    def __init__(self, n_factors=10, learning_rate=0.01, reg=0.1):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.u_name = None
        self.r_name = None

    def _create_mappings(self, df, user_col_name, item_col_name):
        self.u_name = user_col_name
        self.i_name = item_col_name
        users = df[self.u_name].unique()
        items = df[item_col_name].unique()
        self.user_to_idx = {uid: i for i, uid in enumerate(users)}
        self.item_to_idx = {iid: i for i, iid in enumerate(items)}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        self.idx_to_item = {i: iid for iid, i in self.item_to_idx.items()}
        self.n_users = len(users)
        self.n_items = len(items)
        self.norm = 2

    def _build_sparse_matrix(self, df, rating_col_name):
        row = df[self.u_name].map(self.user_to_idx)
        col = df[self.i_name].map(self.item_to_idx)
        self.r_name = rating_col_name
        data = df[rating_col_name]/self.norm
        return csr_matrix((data, (row, col)), shape=(self.n_users, self.n_items))

    def fit(self, df, user_col_name, item_col_name, rating_col_name='rating', test_df=None, n_iters=10, flag_print=True):
        self._create_mappings(df, user_col_name, item_col_name)
        self.R = self._build_sparse_matrix(df, rating_col_name)

        self.P = np.random.normal(scale=1./self.n_factors, size=(self.n_users, self.n_factors))
        self.Q = np.random.normal(scale=1./self.n_factors, size=(self.n_items, self.n_factors))

        rows, cols = self.R.nonzero()

        for it in range(n_iters):
            for u, i in zip(rows, cols):
                r_ui = self.R[u, i]
                pred = np.dot(self.P[u, :], self.Q[i, :])
                err = r_ui - pred

                self.P[u, :] += self.lr * (err * self.Q[i, :] - self.reg * self.P[u, :])
                self.Q[i, :] += self.lr * (err * self.P[u, :] - self.reg * self.Q[i, :])

            if flag_print and not (test_df is None):
                rmse = self.evaluate(test_df)
                print(f"Итерация {it+1}/{n_iters}, RMSE = {rmse:.4f}")

    def predict_single(self, user_id, item_id):
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return 0
        u = self.user_to_idx[user_id]
        i = self.item_to_idx[item_id]
        return (self.P[u, :] @ self.Q[i, :].T) * self.norm

    def predict_df(self, df):
        if self.u_name is None or self.i_name is None:
            return "Сначала сделайте fit"
        return df.apply(lambda row: self.predict_single(row[self.u_name], row[self.i_name]), axis=1)

    def evaluate(self, df):
        if self.u_name is None or self.i_name is None or self.r_name is None:
            return "Сначала сделайте fit"
        y_true = df[self.r_name].values
        y_pred = self.predict_df(df)
        return np.sqrt(mean_squared_error(y_true, y_pred))
