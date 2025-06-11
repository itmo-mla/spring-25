from datetime import datetime as time

import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm


class LFM:
    def __init__(self, n_clients, n_objects, n_factors, lr, reg, rating_interval):
        self.n_clients = n_clients
        self.n_objects = n_objects
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg

        self.p_ut = np.random.normal(size=(n_clients, n_factors))
        self.q_it = np.random.normal(size=(n_objects, n_factors))

        self.r_i_mean = np.random.uniform(*rating_interval, size=(n_objects, 1))
        self.r_u_mean = np.random.uniform(*rating_interval, size=(n_clients, 1))


    def fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame, epochs):
        logged_metrics = {
            "train_rmse": [], "train_mae": [],
            "eval_rmse": [], "eval_mae": [],
            "fitting_time_ms": 0 
        }

        for e in range(epochs):
            t1 = time.now()
            for _, row in tqdm(train_data.sample(frac=1).iterrows(), 
                               total=len(train_data),
                               desc=f"Epoch {e+1}"):
                u = int(row['userIdx'])
                i = int(row['movieIdx'])

                r_ui = row['rating']
                pred = self.r_i_mean[i] + self.r_u_mean[u] + self.p_ut[u] @ self.q_it[i].T

                error = r_ui - pred 

                self.p_ut[u] += self.lr * (error * self.q_it[i] - self.reg * self.p_ut[u])
                self.q_it[i] += self.lr * (error * self.p_ut[u] - self.reg * self.q_it[i])

                self.r_u_mean[u] += self.lr * (error - self.reg * self.r_u_mean[u])
                self.r_i_mean[i] += self.lr * (error - self.reg * self.r_i_mean[i])

            logged_metrics['fitting_time_ms'] += (time.now() - t1).microseconds / 1000
            
            train_rmse, train_mae = self.eval(train_data)
            eval_rmse, eval_mae = self.eval(test_data)

            logged_metrics['train_rmse'] += [train_rmse]
            logged_metrics['train_mae'] += [train_mae]
            logged_metrics['eval_rmse'] += [eval_rmse]
            logged_metrics['eval_mae'] += [eval_mae]

            print(f"Train | RMSE={train_rmse:.3f} | MAE={train_mae:.3f}" \
                  f" ||| Test | RMSE={eval_rmse:.3f} | MAE={eval_mae:.3f}")
            
        return logged_metrics

    def predict(self, userIdx, movieIdx):
        return self.r_i_mean[movieIdx] + self.r_u_mean[userIdx] + self.p_ut[userIdx] @ self.q_it[movieIdx].T

    def eval(self, data: pd.DataFrame):
        preds = []
        gt = []

        for _, row in data.iterrows():
            u = int(row['userIdx'])
            i = int(row['movieIdx'])

            gt += [row['rating'].astype(np.float32)]
            preds += [(self.r_i_mean[i] + self.r_u_mean[u] + self.p_ut[u] @ self.q_it[i].T).astype(np.float32)]

        rmse = np.sqrt(sklearn.metrics.mean_squared_error(gt, preds))
        mae = sklearn.metrics.mean_absolute_error(gt, preds)

        return rmse, mae