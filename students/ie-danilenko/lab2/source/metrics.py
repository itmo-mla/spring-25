from model import GradientBoostingRegressor, KFold
from sklearn.ensemble import GradientBoostingRegressor as SKBoost
import numpy as np
from sklearn.metrics import mean_squared_error
from read import read_data
from sklearn.model_selection import train_test_split
from time import time

if __name__ == "__main__":
    X, y = read_data("data/possum.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        GradientBoostingRegressor(n_estimators=100, max_depth=16, learning_rate=0.005),
        SKBoost(n_estimators=100, max_depth=16, learning_rate=0.005)
    ]

    fold = KFold()

    metrics = []
    times = []
    for model in models:
        start_time = time()
        metrics.append(fold.cros_valid(model, X_train, y_train, mean_squared_error))
        times.append(time() - start_time)

    print(f"My time: {times[0]}")
    print(f"My metrics: {np.mean(metrics[0])}\n")

    print(f"SKBoost time: {times[1]}")
    print(f"SKBoost metrics: {np.mean(metrics[1])}")