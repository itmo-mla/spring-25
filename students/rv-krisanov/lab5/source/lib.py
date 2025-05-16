import scipy
import numpy as np
import tqdm


class CollaborateFiltrator:  # latent factor model
    def fit(
        self,
        R: scipy.sparse.csr_matrix,
        R_test: scipy.sparse.csr_matrix,
        k: int = 13,
        epochs: int = 20,
    ) -> np.ndarray:
        m, n = R.shape

        rows, cols = R.nonzero()

        eta, lambda_ = 0.01, 0.1

        P = np.random.rand(m, k) * 0.01
        Q = np.random.rand(n, k) * 0.01
        e: scipy.sparse.csr_matrix = R.copy()

        for epoch in range(epochs):
            with tqdm(
                total=len(rows),
                desc=f"Epoch {epoch + 1}/{epochs}",
                ncols=150,
                colour="cyan",
                leave=True,
            ) as epoch_bar:
                for user_id, object_id in zip(rows, cols):
                    e[user_id, object_id] = R[user_id, object_id] - np.dot(
                        P[user_id, :], Q[object_id, :]
                    )

                    p, q = P[user_id, :].copy(), Q[object_id, :].copy()

                    P[user_id, :] += eta * (e[user_id, object_id] * q - lambda_ * p)
                    Q[object_id, :] += eta * (e[user_id, object_id] * p - lambda_ * q)
                    epoch_bar.update(1)
                else:
                    loss = (e.multiply(e)).sum() + (
                        np.linalg.norm(P) ** 2 + np.linalg.norm(Q) ** 2
                    )
                    self.P = P
                    self.Q = Q

                    r_pred = self.predict(*R_test.nonzero())

                    mae = (R_test - r_pred).__abs__().sum() / (R.count_nonzero())
                    rmse = np.sqrt(
                        (R_test - r_pred).multiply(R_test - r_pred).sum()
                        / R.count_nonzero()
                    )

                    epoch_bar.set_postfix(
                        {
                            "loss": f"{loss:.6f}",
                            "MAE": f"{mae:.6f}",
                            "RMSE": f"{rmse:.6f}",
                        }
                    )
                    epoch_bar.update(0)

        self.P = P
        self.Q = Q
        self.m = m
        self.user_count = self.m
        self.n = n
        self.object_count = self.n
        self.k = k
        self.latent_space = self.k

    def predict(
        self, user_ids: list[int] | np.ndarray, object_ids: list[int] | np.ndarray
    ) -> np.ndarray:
        score = np.zeros(shape=(len(user_ids),), dtype=np.float32)
        for idx, [user_id, object_id] in enumerate(zip(user_ids, object_ids)):
            score[idx] = np.dot(self.P[user_id], self.Q[object_id])
        return scipy.sparse.csr_matrix(
            (
                score,
                (
                    np.array(user_ids, dtype=np.int32),
                    np.array(object_ids, dtype=np.int32),
                ),
            ),
            dtype=np.float32,
        )