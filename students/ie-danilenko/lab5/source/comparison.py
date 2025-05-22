from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from read import read_movies
from sklearn.model_selection import train_test_split
from lfm import LFM
from metrics import mae, rmse
import numpy as np
from time import time

if __name__ == "__main__":
    num_factors = 37
    learning_rate = 0.01
    regularization = 0.01
    epochs = 10

    data = read_movies('data/ratings.csv')
    reader = Reader(rating_scale=(1, 5))
    data_surp = Dataset.load_from_df(data[["userId", "movieId", "rating"]], reader)

    model_surprise = SVD(n_factors=num_factors,biased=True, lr_all=learning_rate, reg_all=regularization, n_epochs=epochs)

    cross_val_surprise = cross_validate(model_surprise, data_surp, measures=["RMSE", "MAE"], cv=2, verbose=False)

    rmse_surprise = cross_val_surprise['test_rmse'].mean()
    mae_surprise = cross_val_surprise['test_mae'].mean()
    time_surprise = np.mean(cross_val_surprise['fit_time'])

    print(f"Surprise RMSE: {rmse_surprise:.4f}")
    print(f"Surprise MAE: {mae_surprise:.4f}")
    print(f"Surprise Time: {time_surprise:.4f}")
    print()

    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

    start = time()
    lfm_model = LFM(num_factors, learning_rate, regularization)
    lfm_model.fit(X_train, epochs)
    time_my = time() - start

    actual_ratings = []
    predicted_ratings = []

    predicted_ratings = lfm_model.predict(X_test)
    actual_ratings = X_test['rating'].values

    valid_predictions_mask = ~np.isnan(predicted_ratings)
    predicted_ratings = predicted_ratings[valid_predictions_mask]
    actual_ratings = actual_ratings[valid_predictions_mask]

    mae_my = mae(actual_ratings, predicted_ratings)
    rmse_my = rmse(actual_ratings, predicted_ratings)

    print(f"My RMSE: {rmse_my:.4f}")
    print(f"My MAE: {mae_my:.4f}")
    print(f"My Time: {time_my:.4f}")
