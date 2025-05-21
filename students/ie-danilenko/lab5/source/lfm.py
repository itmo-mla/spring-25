import numpy as np
import pandas as pd
import random

class LFM:
    def __init__(self, num_factors=5, learning_rate=0.01, regularization=0.01):
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.regularization = regularization

    def fit(self, X, epochs):
        self.users_categorical = X['userId'].astype('category')
        self.items_categorical = X['movieId'].astype('category')
        users = self.users_categorical.cat.codes
        items = self.items_categorical.cat.codes
        ratings = X['rating']

        self.num_users = len(self.users_categorical.cat.categories)
        self.num_items = len(self.items_categorical.cat.categories)

        self.user_factors = np.random.rand(self.num_users, self.num_factors)
        self.item_factors = np.random.rand(self.num_items, self.num_factors)

        for index in X.index:
            user_id = users.loc[index]
            item_id = items.loc[index]
            rating = ratings.loc[index]

            prediction = np.dot(self.user_factors[user_id], self.item_factors[item_id])
            error = rating - prediction

            self.user_factors[user_id] += self.learning_rate * (error * self.item_factors[item_id] - self.regularization * self.user_factors[user_id])
            self.item_factors[item_id] += self.learning_rate * (error * self.user_factors[user_id] - self.regularization * self.item_factors[item_id])

    def predict(self, X_test):
        predicted_ratings = []
        
        users_categories = self.users_categorical.cat.categories
        items_categories = self.items_categorical.cat.categories

        for _, row in X_test.iterrows():
            original_user_id = row['userId']
            original_item_id = row['movieId']

            try:
                user_idx = users_categories.get_loc(original_user_id)
                item_idx = items_categories.get_loc(original_item_id)
                
                if user_idx < self.num_users and item_idx < self.num_items:
                    prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                    predicted_ratings.append(prediction)
                else:
                    predicted_ratings.append(np.nan)
            except KeyError:
                predicted_ratings.append(np.nan)

        return np.array(predicted_ratings)
                

if __name__ == "__main__":
    from read import read_movies

    data = read_movies('data/ratings.csv')
    
    num_factors = 10
    learning_rate = 0.01
    regularization = 0.01
    epochs = 10

    lfm_model = LFM(num_factors, learning_rate, regularization)
    
    lfm_model.fit(data, epochs)

    random_row_index = random.choice(data.index)
    example_dat = data.loc[[random_row_index]][["userId", "movieId"]]

    predicted_rating = lfm_model.predict(example_dat)
    print(f"Предсказание рейтинга для пользователя (ориг. ID: {example_dat['userId'].iloc[0]}, Ориг. Movie ID: {example_dat['movieId'].iloc[0]}): {predicted_rating}")