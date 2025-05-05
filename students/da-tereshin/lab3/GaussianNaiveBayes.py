import numpy as np
import pandas as pd


class GaussianNaiveBayes:
    def __init__(self) -> None:
        self.classes = None
        self.means = None
        self.vars = None
        self.priors = None

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        X = x if isinstance(x, np.ndarray) else x.to_numpy()
        Y = y if isinstance(y, np.ndarray) else y.to_numpy()

        self.classes = np.unique(Y)

        self.means = {}
        self.vars = {}
        self.priors = {}
        # Вычисляем средние, дисперсии и априорные вероятности для каждого класса
        for cls in self.classes:
            X_cls = X[Y == cls]
            self.means[cls] = np.mean(X_cls, axis=0)
            self.vars[cls] = np.var(X_cls, axis=0)
            self.priors[cls] = X_cls.shape[0] / X.shape[0]

    def _calculate_likelihood(self, cls: int, x: pd.DataFrame | np.ndarray) -> float:
        X = x if isinstance(x, np.ndarray) else x.to_numpy()

        # Получаем параметры для класса
        mean = self.means[cls]
        var = self.vars[cls]
        # Вычисляем вероятность признака при условии класса
        exponenta = np.exp(-((X - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return exponenta / denominator

    def _calculate_posterior(self, x: pd.DataFrame | np.ndarray):
        posteriors = []
        # Вычисляем апостериорные вероятности для каждого класса
        for cls in self.classes:
            # Переход к сумме логарифмов для численной стабильности
            prior = np.log(self.priors[cls])
            likelihood = np.sum(np.log(self._calculate_likelihood(cls, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        X = x if isinstance(x, np.ndarray) else x.to_numpy()

        if self.classes is None:
            raise ValueError("Модель не обучена")

        y_pred = [self._calculate_posterior(X[i]) for i in range(X.shape[0])]
        return np.array(y_pred)

# %%
# t = np.array([[2, 1, 3], [1, 3, 5]])
# print(t)
# for el in t:
#     print(t)
