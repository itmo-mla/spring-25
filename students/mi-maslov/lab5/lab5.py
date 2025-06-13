# pip install scikit-surprise==1.1.1 numpy==1.23.5 scipy==1.9.3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surprise.model_selection import cross_validate, train_test_split
import time

from os import terminal_size
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from numpy.linalg import norm

class LFM:
    def __init__(self, K=3, lamda=0.02, gamma=0.001, steps=100):
        self.K = K
        self.lamda = lamda
        self.gamma = gamma
        self.steps = steps

    def fit(self, trainset):
        self.trainset = trainset
        self.n_users = trainset.n_users
        self.n_items = trainset.n_items

        ratings_data = []
        rows = []
        cols = []

        for u, i, r in trainset.all_ratings():
            ratings_data.append(r)
            rows.append(u)
            cols.append(i)

        self.R = coo_matrix((ratings_data, (rows, cols)), shape=(self.n_users, self.n_items))
        M, N = self.R.shape

        P = np.random.rand(M, self.K)
        Q = np.random.rand(self.K, N)

        rmse = np.sqrt(self._error(self.R, P, Q, self.lamda) / len(self.R.data))

        for step in range(self.steps):
            for ui in range(len(self.R.data)):
                rui = self.R.data[ui]
                u = self.R.row[ui]
                i = self.R.col[ui]
                if rui > 0:
                    eui = rui - np.dot(P[u,:], Q[:,i])
                    P[u,:] = P[u,:] + self.gamma * 2 * (eui * Q[:,i] - self.lamda * P[u,:])
                    Q[:,i] = Q[:,i] + self.gamma * 2 * (eui * P[u,:] - self.lamda * Q[:,i])

            rmse = np.sqrt(self._error(self.R, P, Q, self.lamda) / len(self.R.data))
            if rmse < 0.5:
                break

        self.P = P
        self.Q = Q

    def _error(self, R, P, Q, lamda):
        ratings = R.data
        rows = R.row
        cols = R.col
        e = 0
        for ui in range(len(ratings)):
            rui = ratings[ui]
            u = rows[ui]
            i = cols[ui]
            if rui > 0:
                e = e + pow(rui - np.dot(P[u,:], Q[:,i]), 2) + \
                    lamda * (pow(norm(P[u,:]), 2) + pow(norm(Q[:,i]), 2))
        return e

    def predict(self):
        all_user_ratings = np.matmul(self.P, self.Q)
        return all_user_ratings

    def predict_rating(self, uid, iid):
        inner_uid = self.trainset.to_inner_uid(uid)
        inner_iid = self.trainset.to_inner_iid(iid)
        return np.dot(self.P[inner_uid], self.Q[:, inner_iid])

    def test(self, testset):
        predictions = []
        for uid, iid, r_ui in testset:
            try:
              pred = self.predict_rating(uid, iid)
              predictions.append((uid, iid, r_ui, pred, abs(r_ui - pred)))
            except:
              continue
        return predictions

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin(name='ml-100k', prompt=False)

trainset, testset = train_test_split(data, test_size=0.25)

start_time = time.time()
lfm = LFM(K=20, lamda=0.01, gamma=0.0007, steps=30)
lfm.fit(trainset)
custom_time = time.time() - start_time

start_time = time.time()
algo = SVD(n_factors=20, n_epochs=30, lr_all=0.01, reg_all=0.001)
algo.fit(trainset)
surprise_time = time.time() - start_time

surprise_predictions = algo.test(testset)
custom_predictions = lfm.test(testset)

def calculate_rmse(predictions):
    squared_errors = [(true_r - est_r) ** 2 for (_, _, true_r, est_r, _) in predictions]
    return np.sqrt(np.mean(squared_errors))

def calculate_mae(predictions):
    abs_errors = [abs(true_r - est_r) for (_, _, true_r, est_r, _) in predictions]
    return np.mean(abs_errors)

def calculate_surprise_rmse(predictions):
    squared_errors = [(pred.r_ui - pred.est) ** 2 for pred in predictions]
    return np.sqrt(np.mean(squared_errors))

def calculate_surprise_mae(predictions):
    abs_errors = [abs(pred.r_ui - pred.est) for pred in predictions]
    return np.mean(abs_errors)

custom_rmse = calculate_rmse(custom_predictions)
custom_mae = calculate_mae(custom_predictions)

surprise_rmse = calculate_surprise_rmse(surprise_predictions)
surprise_mae = calculate_surprise_mae(surprise_predictions)

print(f"Custom LFM - Training Time: {custom_time:.2f}s, RMSE: {custom_rmse:.4f}, MAE: {custom_mae:.4f}")
print(f"Surprise SVD - Training Time: {surprise_time:.2f}s, RMSE: {surprise_rmse:.4f}, MAE: {surprise_mae:.4f}")

import random

def show_top_predictions(predictions, model_name, uid=None):
    if uid is None:
        if model_name == "Custom":
            all_uids = set(p[0] for p in predictions)
        else:
            all_uids = set(p.uid for p in predictions)
        uid = list(all_uids)[12]
        print(f"Randomly selected User {uid} for recommendation.\n")

    print(f"Top 10 Predictions for User {uid} using {model_name} model:\n{'-' * 40}")

    if model_name == "Custom":
        user_preds = [p for p in predictions if p[0] == uid]
        sorted_preds = sorted(user_preds, key=lambda x: x[3], reverse=True)[:10]
        for pred in sorted_preds:
            _, iid, true_r, pred_r, _ = pred
            print(f"Item {iid}: Predicted {pred_r:.2f}, Actual {true_r}")
    else:
        user_preds = [p for p in predictions if p.uid == uid]
        sorted_preds = sorted(user_preds, key=lambda x: x.est, reverse=True)[:10]
        for pred in sorted_preds:
            print(f"Item {pred.iid}: Predicted {pred.est:.2f}, Actual {pred.r_ui}")

show_top_predictions(custom_predictions, "Custom")

show_top_predictions(surprise_predictions, "Surprise")