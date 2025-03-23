import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        random_state: int | None = None,
    ):
        """
        Инициализация градиентного бустинга для задачи регрессии

        Args:
            n_estimators: количество базовых моделей (деревьев)
            learning_rate: скорость обучения (темп)
            max_depth: максимальная глубина деревьев
            subsample: доля выборки для каждого дерева (случайная подвыборка)
            random_state: фиксация генератора случайных чисел
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state

        # Список для хранения базовых моделей
        self.trees: list[DecisionTreeRegressor] = []
        # Начальное значение предсказания (среднее по выборке)
        self.initial_prediction: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        """
        Обучение модели градиентного бустинга

        Args:
            X: матрица признаков
            y: целевая переменная

        Returns:
            self: обученная модель
        """
        # Инициализация генератора случайных чисел
        rng = np.random.RandomState(self.random_state)

        # Инициализация начального предсказания как среднего значения
        self.initial_prediction = np.mean(y)
        F = np.full_like(y, self.initial_prediction, dtype=np.float64)

        # Последовательное обучение деревьев
        for _ in range(self.n_estimators):
            # Вычисление псевдоостатков (градиентов)
            residuals = y - F

            # Создание подвыборки, если subsample < 1
            if self.subsample < 1.0:
                sample_mask = rng.rand(len(y)) < self.subsample
                subsample_X = X[sample_mask]
                subsample_residuals = residuals[sample_mask]
            else:
                subsample_X = X
                subsample_residuals = residuals

            # Создание и обучение нового дерева на псевдоостатках
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, random_state=self.random_state
            )
            tree.fit(subsample_X, subsample_residuals)

            # Добавление дерева в список
            self.trees.append(tree)

            # Обновление текущих предсказаний
            F += self.learning_rate * tree.predict(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Получение предсказаний модели

        Args:
            X: матрица признаков

        Returns:
            predictions: массив предсказаний
        """
        # Проверка, что модель обучена
        if not self.trees or self.initial_prediction is None:
            raise ValueError("Model must be fitted before making predictions")

        # Начальное предсказание
        predictions = np.full(X.shape[0], self.initial_prediction, dtype=np.float64)

        # Добавление вклада каждого дерева
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions
