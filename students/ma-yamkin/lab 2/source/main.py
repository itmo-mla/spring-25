from Loader import Loader
from Model import CrossValidation
import time
from Model import GBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


def time_of_method(func):
    def wrapper():
        start = time.time()
        func()
        end = time.time()
        delta = end - start
        print(f'{delta}')
    return wrapper


@time_of_method
def hand():
    model = GBMRegressor()
    cross_validator = CrossValidation(n_splits=5)

    mae = cross_validator.eval(X, y, model, mean_absolute_error)
    mse = cross_validator.eval(X, y, model, mean_squared_error)

    print(f"MAE: {mae.mean():.2f}")
    print(f"MSE: {mse.mean():.2f}")


@time_of_method
def lib():
    model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=12345)
    cross_validator = CrossValidation(n_splits=5)

    mae = cross_validator.eval(X, y, model, mean_absolute_error)
    mse = cross_validator.eval(X, y, model, mean_squared_error)

    print(f"MAE: {abs(mae.mean()):.2f}")
    print(f"MSE: {abs(mse.mean()):.2f}")


loader = Loader()
X, y = loader.X, loader.y

hand()
print()
lib()
