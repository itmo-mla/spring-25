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

## Решение

В качестве датасета выбран `diabetes` из SKLearn.

### Реализация градиентного бустинга

```python
import numpy as np

from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
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

### Сравнение с эталонной реализацией

| Показатель | Sklearn   | Custom    |
|------------|-----------|-----------|
| MSE        | 3236.9647 | 3236.9647 |
| Время      | 0.0724    | 0.0632    |

### Выводы

Реализация SKlearn и собственная показала идентичные результаты по MSE и даже выиграла по скорости
