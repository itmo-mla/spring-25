from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from collections import defaultdict


class FeatureDistribution(ABC):
    """Абстрактный класс для распределения признака"""

    @abstractmethod
    def fit(self, X: NDArray, y: NDArray) -> None:
        """Обучить параметры распределения"""
        pass

    @abstractmethod
    def probability(self, x: NDArray) -> NDArray:
        """Вычислить вероятность P(x|y) для всех классов"""
        pass


class GaussianDistribution(FeatureDistribution):
    """Нормальное распределение для непрерывных признаков"""

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.mean: dict[int, float] = {}
        self.var: dict[int, float] = {}
        self.var_smoothing = var_smoothing

    def fit(self, X: NDArray, y: NDArray) -> None:
        for class_label in np.unique(y):
            self.mean[class_label] = np.mean(X[y == class_label])
            # Добавляем сглаживание дисперсии как в sklearn
            self.var[class_label] = np.var(X[y == class_label]) + self.var_smoothing

    def probability(self, x: NDArray) -> NDArray:
        probs = np.zeros(len(self.mean))
        for class_label in self.mean:
            diff = x - self.mean[class_label]
            log_prob = -0.5 * (np.log(2 * np.pi * self.var[class_label]) + 
                              (diff ** 2) / self.var[class_label])
            probs[class_label] = np.exp(log_prob)
        return probs


class CategoricalDistribution(FeatureDistribution):
    """Категориальное распределение с лапласовым сглаживанием"""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.probs: dict[int, dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.classes: set[int] = set()
        self.categories: set[int] = set()

    def fit(self, X: NDArray, y: NDArray) -> None:
        self.classes = set(np.unique(y))
        self.categories = set(np.unique(X))

        counts = defaultdict(lambda: defaultdict(int))
        for x_i, y_i in zip(X, y):
            counts[y_i][x_i] += 1

        # Применяем сглаживание Лапласа
        for class_label in self.classes:
            total = sum(counts[class_label].values()) + self.alpha * len(
                self.categories
            )
            for category in self.categories:
                self.probs[class_label][category] = (
                    counts[class_label][category] + self.alpha
                ) / total

    def probability(self, x: NDArray) -> NDArray:
        probs = np.zeros(len(self.classes))
        for i, class_label in enumerate(self.classes):
            probs[i] = self.probs[class_label][x]
        return probs


class BernoulliDistribution(FeatureDistribution):
    """Распределение Бернулли для бинарных признаков"""

    def __init__(self, alpha: float = 1.0) -> None:
        self.probs: dict[int, float] = {}
        self.alpha = alpha  # Добавляем сглаживание Лапласа

    def fit(self, X: NDArray, y: NDArray) -> None:
        for class_label in np.unique(y):
            # Добавляем сглаживание Лапласа
            n_samples = np.sum(y == class_label)
            n_positive = np.sum(X[y == class_label])
            self.probs[class_label] = (n_positive + self.alpha) / (n_samples + 2 * self.alpha)

    def probability(self, x: NDArray) -> NDArray:
        probs = np.zeros(len(self.probs))
        for class_label in self.probs:
            p = self.probs[class_label]
            if x == 1:
                probs[class_label] = p
            else:
                probs[class_label] = 1 - p
        return probs


@dataclass
class NaiveBayesClassifier:
    """Наивный байесовский классификатор с поддержкой разных распределений признаков"""

    feature_distributions: list[FeatureDistribution]
    class_priors: dict[int, float] = field(default_factory=dict)

    def fit(self, X: NDArray, y: NDArray) -> None:
        """
        Обучить классификатор

        Args:
            X: массив признаков размера (n_samples, n_features)
            y: массив меток классов размера (n_samples,)
        """
        # Вычисляем априорные вероятности классов с небольшим сглаживанием
        unique_classes = np.unique(y)
        n_samples = len(y)
        alpha = 1e-10  # сглаживание для априорных вероятностей
        
        for class_label in unique_classes:
            count = np.sum(y == class_label)
            self.class_priors[class_label] = (count + alpha) / (n_samples + alpha * len(unique_classes))

        # Обучаем распределения для каждого признака
        for feature_idx, distribution in enumerate(self.feature_distributions):
            distribution.fit(X[:, feature_idx], y)

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Вычислить вероятности классов для входных данных

        Args:
            X: массив признаков размера (n_samples, n_features)

        Returns:
            Массив вероятностей классов размера (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        n_classes = len(self.class_priors)
        log_probs = np.zeros((n_samples, n_classes))

        # Добавляем логарифм априорных вероятностей
        for class_idx, prior in self.class_priors.items():
            log_probs[:, class_idx] = np.log(prior)

        # Добавляем логарифм правдоподобия для каждого признака
        for sample_idx in range(n_samples):
            for feature_idx, distribution in enumerate(self.feature_distributions):
                feature_probs = distribution.probability(X[sample_idx, feature_idx])
                # Используем более стабильный способ работы с логарифмами
                with np.errstate(divide='ignore'):  # игнорируем предупреждения о log(0)
                    log_feature_probs = np.log(feature_probs + 1e-300)
                log_probs[sample_idx, :] += log_feature_probs

        # Нормализуем вероятности более стабильным способом
        log_prob_max = log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs - log_prob_max)
        probs_sum = probs.sum(axis=1, keepdims=True)
        return probs / probs_sum

    def predict(self, X: NDArray) -> NDArray:
        """
        Предсказать метки классов для входных данных

        Args:
            X: массив признаков размера (n_samples, n_features)

        Returns:
            Массив предсказанных меток классов размера (n_samples,)
        """
        return np.argmax(self.predict_proba(X), axis=1)
