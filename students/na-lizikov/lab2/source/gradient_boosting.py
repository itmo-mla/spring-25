import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from typing import Optional, Union, List

class GradientBoosting:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: Optional[int] = None
    ):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        self.trees: List[DecisionTreeRegressor] = []
        self.F0: Optional[float] = None

    def _initialize_prediction(self, y: np.ndarray) -> float:
        return np.mean(y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoosting':
        self.F0 = self._initialize_prediction(y)
        F = np.full(len(y), self.F0)
        
        for _ in range(self.n_estimators):
            # Вычисление градиентов (остатков)
            residuals = y - F
            
            # Создание и обучение нового дерева
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )
            tree.fit(X, residuals)
            
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            self.trees.append(tree)
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.F0 is None:
            raise ValueError("Модель не обучена. Сначала выполните fit().")
            
        # Начальное предсказание
        predictions = np.full(len(X), self.F0)
        
        # Добавление вклада каждого дерева
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
            
        return predictions 