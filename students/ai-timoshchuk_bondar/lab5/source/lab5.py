import numpy as np
from collections import defaultdict
from typing import Iterable, Tuple, List
import time, math, random
from tqdm import tqdm
from surprise import Dataset, SVD, Reader, accuracy, Trainset
from surprise.model_selection import cross_validate, train_test_split


Rating = Tuple[int, int, float]  # (user_id, item_id, rating)


class LatentFactorModel:
    def __init__(
        self,
        n_factors: int = 50,
        lr: float = 0.01,
        reg: float = 0.02,
        n_epochs: int = 20,
        seed: int | None = 42,
    ) -> None:
        self.n_factors, self.lr, self.reg, self.n_epochs = n_factors, lr, reg, n_epochs
        self.rng = np.random.default_rng(seed)

    def fit(self, ratings: Iterable[Rating]) -> "LatentFactorModel":
        self._build_mappings(ratings)
        self._init_params()

        data = [(self.u2i[u], self.i2i[i], r) for u, i, r in ratings]
        for epoch in range(self.n_epochs):
            self.rng.shuffle(data)
            self._sgd_epoch(data)
            rmse = self._rmse(data)
            print(f"Epoch {epoch + 1:02d}/{self.n_epochs}  RMSE = {rmse:.4f}")
        return self

    def predict_single(self, user: int, item: int) -> float:
        """Предсказание для пары (user_id, item_id)."""
        if user not in self.u2i or item not in self.i2i:
            return self.global_mean  # максимально простая обработка cold-start
        u, i = self.u2i[user], self.i2i[item]
        return self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])

    def recommend(self, user: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Top рекомендаций для пользователя (возвращает item_id и скор)."""
        if user not in self.u2i:
            raise ValueError("Unknown user")
        u = self.u2i[user]
        scores = self.global_mean + self.bi + self.bu[u] + self.Q @ self.P[u]
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        return [
            (self.i2id[i], float(scores[i]))
            for i in top_idx[np.argsort(-scores[top_idx])]
        ]

    def _build_mappings(self, ratings: Iterable[Rating]) -> None:
        users, items = set(), set()
        for u, i, _ in ratings:
            users.add(u)
            items.add(i)
        self.u2i = {u: idx for idx, u in enumerate(sorted(users))}
        self.i2u = {idx: u for u, idx in self.u2i.items()}
        self.i2i = {i: idx for idx, i in enumerate(sorted(items))}
        self.i2id = {idx: i for i, idx in self.i2i.items()}
        self.n_users, self.n_items = len(self.u2i), len(self.i2i)

    def _init_params(self) -> None:
        self.P = 0.01 * self.rng.standard_normal((self.n_users, self.n_factors))
        self.Q = 0.01 * self.rng.standard_normal((self.n_items, self.n_factors))
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)
        self.global_mean = 0.0

    def _sgd_epoch(self, data: List[Tuple[int, int, float]]) -> None:
        μgrad = 0.0
        for u, i, r in data:
            pred = (
                self.global_mean
                + self.bu[u]
                + self.bi[i]
                + np.dot(self.P[u], self.Q[i])
            )
            err = r - pred

            # градиентный шаг
            self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
            self.Q[i] += self.lr * (err * self.P[u] - self.reg * self.Q[i])
            self.bu[u] += self.lr * (err - self.reg * self.bu[u])
            self.bi[i] += self.lr * (err - self.reg * self.bi[i])
            μgrad += err

        # глобальное смещение (можно обновлять реже, но здесь раз в эпоху)
        self.global_mean += self.lr * μgrad / len(data)

    def _rmse(self, data: List[Tuple[int, int, float]]) -> float:
        se = 0.0
        for u, i, r in data:
            pred = (
                self.global_mean
                + self.bu[u]
                + self.bi[i]
                + np.dot(self.P[u], self.Q[i])
            )
            se += (r - pred) ** 2
        return np.sqrt(se / len(data))


# 1. Загружаем MovieLens-100K и конвертируем в список троек (user, item, rating)
data = Dataset.load_builtin("ml-100k")
full_trainset: Trainset = data.build_full_trainset()

triples = [
    (full_trainset.to_raw_uid(u), full_trainset.to_raw_iid(i), r)
    for (u, i, r) in full_trainset.all_ratings()
]

random.seed(42)
random.shuffle(triples)
split = int(0.75 * len(triples))
trainset, testset = train_test_split(data, test_size=0.25)
train_triples, test_triples = triples[:split], triples[split:]


print("\n=== NumPy LatentFactorModel ===")
lfm = LatentFactorModel(n_factors=50, lr=0.01, reg=0.02, n_epochs=30)

start = time.perf_counter()
lfm.fit(train_triples)
train_time = time.perf_counter() - start

# RMSE на тесте
se = 0.0
for u, i, r in test_triples:
    se += (r - lfm.predict_single(u, i)) ** 2
rmse_lfm = math.sqrt(se / len(test_triples))


start = time.perf_counter()
for u, i, _ in test_triples:
    _ = lfm.predict_single(u, i)
infer_time = (time.perf_counter() - start) / len(test_triples)

print(
    f"train_time = {train_time:.2f} s,   RMSE = {rmse_lfm:.4f},   "
    f"mean_pred_time = {infer_time*1e6:.1f} µs"
)


print("\n=== scikit-surprise SVD ===")
svd = SVD(
    n_factors=50,
    lr_all=0.01,
    reg_all=0.02,
    n_epochs=30,
    random_state=42,
)

start = time.perf_counter()
svd.fit(trainset)
train_time_svd = time.perf_counter() - start

preds = svd.test(testset)
rmse_svd = accuracy.rmse(preds, verbose=False)

start = time.perf_counter()
for uid, iid, _ in test_triples:
    svd.predict(uid, iid, verbose=False)
infer_time_svd = (time.perf_counter() - start) / len(test_triples)

print(
    f"train_time = {train_time_svd:.2f} s,   RMSE = {rmse_svd:.4f},   "
    f"mean_pred_time = {infer_time_svd*1e6:.1f} µs"
)


print("\n=== SUMMARY ===")
print(f"{'Model':<20}{'Train, s':>10}{'Pred, µs':>12}{'RMSE':>10}")
print(f"{'NumPy LFM':<20}{train_time:>10.2f}{infer_time*1e6:>12.1f}{rmse_lfm:>10.4f}")
print(
    f"{'Surprise SVD':<20}{train_time_svd:>10.2f}{infer_time_svd*1e6:>12.1f}{rmse_svd:>10.4f}"
)
