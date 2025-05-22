import numpy as np

def mae(true_ratings, predicted_ratings):
    true_ratings = np.array(true_ratings)
    predicted_ratings = np.array(predicted_ratings)
    return np.mean(np.abs(true_ratings - predicted_ratings))

def rmse(true_ratings, predicted_ratings):
    true_ratings = np.array(true_ratings)
    predicted_ratings = np.array(predicted_ratings)
    return np.sqrt(np.mean((true_ratings - predicted_ratings)**2))
