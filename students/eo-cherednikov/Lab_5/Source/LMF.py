import numpy as np
from typing import Tuple, List

RatingTriplet = Tuple[int, int, float]


class LatentFactorModel:
    def __init__(self, n_users: int, n_items: int, n_factors: int = 40):
        self.user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = 0.0


    def _predict_single(self, user_idx: int, item_idx: int) -> float:
        return (
                self.global_bias
                + self.user_bias[user_idx]
                + self.item_bias[item_idx]
                + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )


    def _update_factors(
            self, user_idx: int, item_idx: int, rating: float, learning_rate: float, reg: float
    ) -> None:
        prediction = self._predict_single(user_idx, item_idx)
        error = rating - prediction

        self.user_bias[user_idx] += learning_rate * (error - reg * self.user_bias[user_idx])
        self.item_bias[item_idx] += learning_rate * (error - reg * self.item_bias[item_idx])

        user_factor = self.user_factors[user_idx].copy()
        item_factor = self.item_factors[item_idx].copy()

        self.user_factors[user_idx] += learning_rate * (error * item_factor - reg * user_factor)
        self.item_factors[item_idx] += learning_rate * (error * user_factor - reg * item_factor)



    def fit(
            self, ratings: List[RatingTriplet], learning_rate: float = 0.01, reg: float = 0.05, n_epochs: int = 15
    ) -> "LatentFactorModel":
        self.global_bias = np.mean([r for _, _, r in ratings])

        for epoch in range(n_epochs):
            np.random.shuffle(ratings)
            epoch_loss = 0.0

            for user_idx, item_idx, rating in ratings:
                user_idx, item_idx = int(user_idx), int(item_idx)
                self._update_factors(user_idx, item_idx, rating, learning_rate, reg)
                prediction = self._predict_single(user_idx, item_idx)
                epoch_loss += (rating - prediction) ** 2

            rmse = np.sqrt(epoch_loss / len(ratings))
            print(f"Epoch {epoch + 1}/{n_epochs} | Train RMSE: {rmse:.4f}")

        return self



    def predict(self, user_idx: int, item_idx: int) -> float:
        return self._predict_single(user_idx, item_idx)