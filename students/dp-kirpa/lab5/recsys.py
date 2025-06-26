import pandas as pd
import numpy as np
import random

seed = 42
np.random.seed(seed)
random.seed(seed)

"""## Данные"""

import pandas as pd
import requests
import zipfile
import io
import os

def download_movielens(size='small', path='./data'):
    if size == 'small':
        url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    elif size == '1m':
        url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    elif size == '25m':
        url = 'https://files.grouplens.org/datasets/movielens/ml-25m.zip'
    elif size == 'latest-full':
        url = 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
    else:
        raise ValueError("Размер должен быть 'small', '1m', '25m' или 'latest-full'")

    if not os.path.exists(path):
        os.makedirs(path)

    print(f"Скачивание {url}...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)
    print(f"Данные сохранены в {path}")

    dataset_dir = [f for f in os.listdir(path) if f.startswith('ml-')][0]
    return os.path.join(path, dataset_dir)

dataset_path = download_movielens('small')

ratings = pd.read_csv(os.path.join(dataset_path, 'ratings.csv'))
movies = pd.read_csv(os.path.join(dataset_path, 'movies.csv'))

print(f"Данные о рейтингах: {ratings.shape}")
print(ratings.head())

print(f"Данные о фильмах: {movies.shape}")
print(movies.head())

from sklearn.model_selection import train_test_split

user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
ratings_matrix = user_movie_matrix.values

X_train, X_test = train_test_split(ratings_matrix, test_size=0.2, random_state=42)

"""# LFM"""

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF

class LatentFactorModel:
    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.02, n_epochs=100, random_state=None):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.user_factors = None
        self.item_factors = None
        self.global_bias = None
        self.user_biases = None
        self.item_biases = None

    def fit(self, X):
        np.random.seed(self.random_state)

        X = np.asarray(X)
        n_users, n_items = X.shape

        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.global_bias = np.mean(X[X > 0])
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

        users, items = np.where(X > 0)
        n_ratings = len(users)
        ratings = X[users, items]

        for epoch in range(self.n_epochs):

            indices = np.arange(n_ratings)
            np.random.shuffle(indices)

            for idx in indices:
                u, i = users[idx], items[idx]
                r = ratings[idx]

                prediction = self.global_bias + self.user_biases[u] + self.item_biases[i] + \
                             np.dot(self.user_factors[u], self.item_factors[i])

                error = r - prediction

                self.user_biases[u] += self.learning_rate * (error - self.regularization * self.user_biases[u])
                self.item_biases[i] += self.learning_rate * (error - self.regularization * self.item_biases[i])

                user_factor = self.user_factors[u].copy()
                item_factor = self.item_factors[i].copy()

                self.user_factors[u] += self.learning_rate * (error * item_factor - self.regularization * self.user_factors[u])
                self.item_factors[i] += self.learning_rate * (error * user_factor - self.regularization * self.item_factors[i])

        return self

    def predict(self, X=None):
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Модель должна быть обучена перед предсказанием.")

        n_users, n_items = len(self.user_biases), len(self.item_biases)

        predictions = np.zeros((n_users, n_items))
        for u in range(n_users):
            for i in range(n_items):
                predictions[u, i] = self.global_bias + self.user_biases[u] + self.item_biases[i] + \
                                   np.dot(self.user_factors[u], self.item_factors[i])

        if X is not None:
            X = np.asarray(X)
            mask = X > 0
            predictions = predictions * mask

        return predictions

    def factorize(self, X):
        self.fit(X)
        return self.user_factors, self.item_factors.T

    def reconstruct(self):
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Модель должна быть обучена перед реконструкцией.")

        return self.predict()


def compare_lfm_with_sklearn(X, n_factors=10, random_state=42):
    X = np.asarray(X)

    X_nmf = X.copy()
    X_nmf[X_nmf == 0] = 0.001

    lfm = LatentFactorModel(n_factors=n_factors, n_epochs=100, random_state=random_state)
    lfm.fit(X)
    X_pred_lfm = lfm.predict()

    nmf = NMF(n_components=n_factors, init='random', random_state=random_state)
    W = nmf.fit_transform(X_nmf)
    H = nmf.components_
    X_pred_nmf = np.dot(W, H)

    mask = X > 0

    rmse_lfm = np.sqrt(mean_squared_error(X[mask], X_pred_lfm[mask]))
    rmse_nmf = np.sqrt(mean_squared_error(X[mask], X_pred_nmf[mask]))

    results = {
        'LFM RMSE': rmse_lfm,
        'NMF RMSE': rmse_nmf,
        'LFM Factors': (lfm.user_factors, lfm.item_factors),
        'NMF Factors': (W, H)
    }

    return results

results = compare_lfm_with_sklearn(X_train, n_factors=2)
print(f"LFM RMSE: {results['LFM RMSE']:.4f}")
print(f"NMF RMSE: {results['NMF RMSE']:.4f}")

lfm = LatentFactorModel(n_factors=20, n_epochs=100, learning_rate=0.01)
lfm.fit(X_train)

predictions = lfm.predict()

predictions