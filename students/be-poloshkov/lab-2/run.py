import time

from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from boosting import GradientBoostingRegressor as MyGradientBoostingRegressor

# 1. Загружаем датасет
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. Смотрим на эталонную реализацию
sklearn_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
start = time.time()
sklearn_regressor.fit(X_train, y_train)
end = time.time()
y_pred = sklearn_regressor.predict(X_test)
print(f'Sklearn MSE: {mean_squared_error(y_test, y_pred):.4f}')
print(f'Sklearn Time: {end - start:.4f}')


# 3. Смотрим на свою реализацию
my_regressor = MyGradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
start = time.time()
my_regressor.fit(X_train, y_train)
end = time.time()
y_pred = my_regressor.predict(X_test)
print(f'My MSE: {mean_squared_error(y_test, y_pred):.4f}')
print(f'My Time: {end - start:.4f}')

