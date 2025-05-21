import pandas as pd
from sklearn.metrics import mean_squared_error
import time
import numpy as np
from sklearn.model_selection import train_test_split
from lightfm import LightFM
from lightfm.datasets import fetch_movielens

# команды для загрузки датасета
# !wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
# !unzip ml-100k.zip

data = pd.read_csv('ml-100k/u.data', sep='\t', header=None,
                   names=['user_id', 'item_id', 'rating', 'timestamp'])

# Преобразование ID пользователей и фильмов в целые числа
user_ids = data['user_id'].unique()
item_ids = data['item_id'].unique()

user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
item_to_idx = {item: idx for idx, item in enumerate(item_ids)}

data['user_idx'] = data['user_id'].map(user_to_idx)
data['item_idx'] = data['item_id'].map(item_to_idx)

# Разделение на train/test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

class LatentFactorModel:
    def __init__(self, num_factors=10, learning_rate=0.01, reg=0.1, epochs=20):
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs

    def fit(self, train_data, num_users, num_items):
        # Инициализация матриц
        self.user_factors = np.random.normal(scale=1./self.num_factors,
                                            size=(num_users, self.num_factors))
        self.item_factors = np.random.normal(scale=1./self.num_factors,
                                            size=(num_items, self.num_factors))

        # Обучение
        for epoch in range(self.epochs):
            for _, row in train_data.iterrows():
                user = row['user_idx']
                item = row['item_idx']
                rating = row['rating']
                pred = self._predict(user, item)
                error = rating - pred

                # Обновление векторов
                self.user_factors[user] += self.learning_rate * (error * self.item_factors[item] - self.reg * self.user_factors[user])
                self.item_factors[item] += self.learning_rate * (error * self.user_factors[user] - self.reg * self.item_factors[item])

    def _predict(self, user, item):
        return np.dot(self.user_factors[user], self.item_factors[item])

    def predict(self, test_data):
        preds = []
        for _, row in test_data.iterrows():
            preds.append(self._predict(row['user_idx'], row['item_idx']))
        return preds

# Обучение модели
num_users = len(user_ids)
num_items = len(item_ids)

lfm = LatentFactorModel(num_factors=20, learning_rate=0.01, reg=0.1, epochs=10)
start_time = time.time()
lfm.fit(train_data, num_users, num_items)
training_time = time.time() - start_time

# Предсказания
test_preds = lfm.predict(test_data)

# Метрики
rmse = np.sqrt(mean_squared_error(test_data['rating'], test_preds))

print(f"RMSE: {rmse:.4f}, Training Time: {training_time:.2f}s")

# Загрузка данных
data = fetch_movielens(genre_features=False, data_home='.', indicator_features=True)

train = data['train']
test = data['test']

# Инициализация модели
model = LightFM(loss='logistic', no_components=32, random_state=42)

# Обучение модели
model.fit(train, epochs=30, num_threads=2)

# Предсказание на тестовой матрице
predicted = model.predict(test.row, test.col)

# Извлечение реальных рейтингов из тестовой матрицы
true_ratings = test.data

# Вычисление RMSE
rmse = np.sqrt(mean_squared_error(true_ratings, predicted))
print(f"RMSE на тестовых данных: {rmse:.4f}")
