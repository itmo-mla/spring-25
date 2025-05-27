import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class GradientBoosting:
    def __init__(self, learning_rate: float = 0.1, n_estimators: int = 10,
                 criterion: str = 'friedman_mse', max_depth: int = 3,
                 random_state: int = 42):
        self.lr = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state

        self.estimators_ = []
        self.initial_prediction_ = None
        self.train_predictions_ = []
        self.test_predictions_ = []

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        # Стираем предыдущие результаты обучения
        if self.estimators_:
            self.estimators_ = []
            self.train_predictions_ = []
            self.test_predictions_ = []

        # Исходный датасет
        X = x if isinstance(x, np.ndarray) else x.to_numpy()
        Y = y if isinstance(y, np.ndarray) else y.to_numpy()

        # Инициализируем начальные предсказания
        self.initial_prediction_ = np.mean(Y)
        # self.initial_prediction_ = 0
        y_pred = np.full_like(Y, self.initial_prediction_, dtype=np.float64)
        # y_pred = np.zeros_like(Y, dtype=np.float64)

        for _ in range(self.n_estimators):
            # Производная mse = разность
            residuals = Y - y_pred

            model = DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth,
                                          random_state=self.random_state)
            model.fit(X, residuals)

            # Обновляем предсказания
            y_pred += self.lr * model.predict(X)
            self.train_predictions_.append(y_pred.copy())

            self.estimators_.append(model)

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("Модель не обучена")
        self.test_predictions_ = []

        X = x if isinstance(x, np.ndarray) else x.to_numpy()

        # Инициализируем предсказания
        # y_pred = np.zeros(X.shape[0])
        y_pred = np.full(X.shape[0], self.initial_prediction_)

        for model in self.estimators_:
            y_pred += self.lr * model.predict(X)
            self.test_predictions_.append(y_pred.copy())

        return y_pred
