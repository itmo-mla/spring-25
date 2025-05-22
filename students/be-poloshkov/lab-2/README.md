# Лабораторная работа №2. Градиентный бустинг

В рамках данной лабораторной работы предстоит реализовать алгоритм градиентного бустинга и сравнить его с эталонной реализацией из библиотеки `scikit-learn`.

## Задание

1. Выбрать датасет для анализа, например, на [kaggle](https://www.kaggle.com/datasets).
2. Реализовать алгоритм градиентного бустинга.
3. Обучить модель на выбранном датасете.
4. Оценить качество модели с использованием кросс-валидации.
5. Замерить время обучения модели.
6. Сравнить результаты с эталонной реализацией из библиотеки [scikit-learn](https://scikit-learn.org/stable/):
   * точность модели;
   * время обучения.
7. Подготовить отчет, включающий:
   * описание алгоритма градиентного бустинга;
   * описание датасета;
   * результаты экспериментов;
   * сравнение с эталонной реализацией;
   * выводы.

## Датасет

Выбран библиотечный diabetes.

## Алгоритм градиентного бустинга
```python
import numpy as np

from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.2, max_depth=4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        current_prediction = np.full(y.shape, fill_value=self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = y - current_prediction

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            predictions_update = tree.predict(X)
            current_prediction += self.learning_rate * predictions_update

    def predict(self, X):
        prediction = np.full(X.shape[0], fill_value=self.initial_prediction)
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)
        return prediction
```
## Тестирование

Представлено в файле `run.py`

```python
# 1. Загружаем датасет
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. Смотрим на эталонную реализацию
sklearn_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=3)
start = time.time()
sklearn_regressor.fit(X_train, y_train)
end = time.time()
y_pred = sklearn_regressor.predict(X_test)
print(f'Sklearn MSE: {mean_squared_error(y_test, y_pred):.4f}')
print(f'Sklearn Time: {end - start:.4f}')


# 3. Смотрим на свою реализацию
my_regressor = MyGradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=3)
start = time.time()
my_regressor.fit(X_train, y_train)
end = time.time()
y_pred = my_regressor.predict(X_test)
print(f'My MSE: {mean_squared_error(y_test, y_pred):.4f}')
print(f'My Time: {end - start:.4f}')
```


### Результаты сравнения

| Показатель | Sklearn   | Self-made |
|------------|-----------|-----------|
| MSE        | 2911.4326 | 2898.1648 |
| Время      | 0.0556    | 0.0596    |

### Выводы

Значения среднеквадратичной ошибки практически идентичны, время работы отличается на тысячные доли секунды
