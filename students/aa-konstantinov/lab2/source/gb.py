import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from sklearn.base import BaseEstimator, RegressorMixin



class CustomGradientBoostingRegressor(RegressorMixin, BaseEstimator):
    """
    Реализация градиентного бустинга для задачи регрессии.
    
    Параметры:
    n_estimators : int, default=100
        Количество деревьев в ансамбле.
    learning_rate : float, default=0.1
        Скорость обучения, которая определяет вклад каждого дерева.
    max_depth : int, default=3
        Максимальная глубина деревьев.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
    
    def fit(self, X, y):
        """
        Обучает модель градиентного бустинга.
        
        Параметры:
        X : array-like, shape (n_samples, n_features)
            Обучающие данные.
        y : array-like, shape (n_samples,)
            Целевые значения.
            
        Возвращает:
        self : object
        """
        X = np.array(X)
        y = np.array(y)
        
        self.initial_prediction = np.mean(y)
        y_pred = np.zeros(y.shape) + self.initial_prediction
        
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            
        return self
    
    def predict(self, X):
        """
        Предсказывает целевые значения для X.
        
        Параметры:
        X : array-like, shape (n_samples, n_features)
            Данные для предсказания.
            
        Возвращает:
        y_pred : array, shape (n_samples,)
            Предсказанные значения.
        """
        if self.initial_prediction is None:
            raise ValueError("Модель не обучена. Вызовите метод fit перед predict.")
            
        X = np.array(X)
        y_pred = np.zeros(X.shape[0]) + self.initial_prediction
        
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
            
        return y_pred
    
    def score(self, X, y):
        """
        Вычисляет коэффициент детерминации R^2 модели.
        
        Параметры:
        X : array-like, shape (n_samples, n_features)
            Данные для предсказания.
        y : array-like, shape (n_samples,)
            Истинные значения.
            
        Возвращает:
        
        score : float
            Коэффициент детерминации R^2.
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v
    




