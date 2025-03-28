import numpy as np
import pandas as pd


class Bagging:
    def __init__(self, estimator, n_estimators: int, max_samples: float = 1.0, random_seed: int = 42) -> None:
        self.base_estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_seed

        self.estimators_ = []

    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(y_pred == y_true)

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        # Стираем предыдущие результаты обучения
        if self.estimators_:
            self.estimators_ = []

        # Исходный датасет
        X = x if isinstance(x, np.ndarray) else x.to_numpy()
        Y = y if isinstance(y, np.ndarray) else y.to_numpy()

        # Фиксируем seed
        rng = np.random.default_rng(seed=self.random_state)

        dataset_full_size = X.shape[0]
        # Размер бутстрэп-выборок
        susbset_size = int(dataset_full_size * self.max_samples)

        # Обучаем ансамбль
        while len(self.estimators_) < self.n_estimators:
            # Создаем бутстрэп-выборку
            indices = rng.choice(a=dataset_full_size, size=susbset_size, replace=True)
            X_subset = X[indices]
            Y_subset = Y[indices]

            # Обучаем базовую модель (создаем новый экземпляр)
            estimator = self.base_estimator.__class__(**self.base_estimator.get_params())
            estimator.fit(X_subset, Y_subset)
            Y_pred = estimator.predict(X_subset)

            if self.accuracy_score(y_true=Y_subset, y_pred=Y_pred) > 0.5:
                self.estimators_.append(estimator)

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("Модель не обучена")

        X = x if type(x) == np.ndarray else x.to_numpy()
        predictions = np.zeros((X.shape[0], self.n_estimators))

        # Получаем предсказания для каждого из алгоритмов
        for i, estimator in enumerate(self.estimators_):
            predictions[:, i] = estimator.predict(X)

        # В результате берем наиболее часто встречающийся класс
        majority_vote = np.apply_along_axis(lambda f: np.bincount(f.astype('int')).argmax(), axis=1, arr=predictions)

        return majority_vote
