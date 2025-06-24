import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class LFM:
    """Модель латентных факторов (LFM) для рекомендательных систем."""

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        """
        Инициализация модели.

        Args:
            n_factors: Количество латентных факторов.
            n_epochs: Количество эпох обучения.
            lr: Скорость обучения (learning rate).
            reg: Коэффициент регуляризации.
            random_state: Для воспроизводимости результатов.
            verbose: Флаг для вывода прогресса обучения.
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        self.verbose = verbose

        self.user_map_: dict[int, int] = {}
        self.item_map_: dict[str, int] = {}
        self.n_users_: int = 0
        self.n_items_: int = 0
        self.global_mean_: float = 0.0
        self.P_: np.ndarray | None = None
        self.Q_: np.ndarray | None = None
        self.bu_: np.ndarray | None = None
        self.bi_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Обучение модели на тренировочных данных.

        Args:
            X: DataFrame с двумя колонками ('User-ID', 'ISBN').
            y: Series с рейтингами ('Rating').

        Returns:
            self: Обученная модель.
        """
        X_train = X.copy()
        X_train["rating"] = y.values

        self.user_map_ = {
            user_id: i for i, user_id in enumerate(X_train["User-ID"].unique())
        }
        self.item_map_ = {
            item_id: i for i, item_id in enumerate(X_train["ISBN"].unique())
        }
        self.n_users_ = len(self.user_map_)
        self.n_items_ = len(self.item_map_)

        X_train["user_idx"] = X_train["User-ID"].map(self.user_map_)
        X_train["item_idx"] = X_train["ISBN"].map(self.item_map_)

        self.global_mean_ = y.mean()

        rng = np.random.default_rng(self.random_state)
        self.P_ = rng.normal(
            0, 1 / self.n_factors, size=(self.n_users_, self.n_factors)
        )
        self.Q_ = rng.normal(
            0, 1 / self.n_factors, size=(self.n_items_, self.n_factors)
        )
        self.bu_ = np.zeros(self.n_users_)
        self.bi_ = np.zeros(self.n_items_)

        iterator = range(self.n_epochs)
        if self.verbose:
            iterator = tqdm(iterator, desc="Обучение LFM")

        for _ in iterator:
            shuffled_X = X_train.sample(frac=1, random_state=rng)
            for _, row in shuffled_X.iterrows():
                self._update_params(row)

        return self

    def _update_params(self, row: pd.Series) -> None:
        """Обновление параметров модели для одной записи."""
        if self.P_ is None or self.Q_ is None or self.bu_ is None or self.bi_ is None:
            return

        u_idx, i_idx = int(row["user_idx"]), int(row["item_idx"])
        rating = row["rating"]

        pred = self._predict_for_indices(u_idx, i_idx)
        error = rating - pred

        self.bu_[u_idx] += self.lr * (error - self.reg * self.bu_[u_idx])
        self.bi_[i_idx] += self.lr * (error - self.reg * self.bi_[i_idx])

        p_u, q_i = self.P_[u_idx, :].copy(), self.Q_[i_idx, :].copy()

        self.P_[u_idx, :] += self.lr * (error * q_i - self.reg * p_u)
        self.Q_[i_idx, :] += self.lr * (error * p_u - self.reg * q_i)

    def _predict_for_indices(self, u_idx: int, i_idx: int) -> float:
        """Предсказание рейтинга по внутренним индексам."""
        if self.P_ is None or self.Q_ is None or self.bu_ is None or self.bi_ is None:
            return self.global_mean_

        user_part = self.bu_[u_idx] + self.P_[u_idx, :] @ self.Q_[i_idx, :].T
        return self.global_mean_ + user_part + self.bi_[i_idx]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание рейтингов для новых данных.

        Args:
            X: DataFrame с колонками ('User-ID', 'ISBN').

        Returns:
            np.ndarray: Массив с предсказанными рейтингами.
        """
        predictions = []
        for _, row in X.iterrows():
            user_id, item_id = row["User-ID"], row["ISBN"]
            u_idx = self.user_map_.get(user_id)
            i_idx = self.item_map_.get(item_id)

            pred = self.global_mean_
            if u_idx is not None and self.bu_ is not None:
                pred += self.bu_[u_idx]
            if i_idx is not None and self.bi_ is not None:
                pred += self.bi_[i_idx]
            if (
                u_idx is not None
                and i_idx is not None
                and self.P_ is not None
                and self.Q_ is not None
            ):
                pred += self.P_[u_idx, :] @ self.Q_[i_idx, :].T
            predictions.append(pred)

        return np.array(predictions)
