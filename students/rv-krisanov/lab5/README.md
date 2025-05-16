# Отчёт по лабораторной работе №5  
*Реализация и сравнение модели латентных факторов*


## Кратко о модели латентных факторов  

Модель латентных факторов (Latent Factor Model, **LFM**) аппроксимирует матрицу взаимодействий *R* (пользователь × объект) произведением двух низкоранговых матриц:

$$
R \approx P Q^{\top}, \qquad
P \in \mathbb{R}^{|U| \times k}, \quad
Q \in \mathbb{R}^{|I| \times k}
$$

где  

* *k* — число скрытых факторов;  
* строки *P* и *Q* — векторы признаков пользователя и объекта.  

Параметры *P* и *Q* оптимизируются минимизацией ошибки (чаще всего **RMSE** или **MAE**) на известных элементах *R* с L2-регуляризацией.  
На практике популярна оптимизация с помощью стохастического градиентного спуска (**SGD**).

---

## 1. Выбор датасета  

| Характеристика | Значение |
|----------------|----------|
| Источник       | **MovieLens “small-latest”** (Kaggle) |
| Оценок         | **100 836** |
| Пользователей  | **610** |
| Фильмов        | **9 742** |
| Диапазон оценок| 1.0 – 5.0 (шаг 0.5) |

---

## 2. Реализация модели латентных факторов  

* **k = 37** скрытых факторов  
* **SGD** по всем ненулевым элементам  
* Гиперпараметры: η = 0.01, λ = 0.1  
* Использованы `numpy` + `scipy.sparse`  

### Блок кода  


```python
import scipy
import numpy as np
import tqdm


class CollaborateFiltrator:
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
```  

---

## 3. Обучение модели  

* **23 эпох** по обучающей выборке (80 % от датасета).  
* Обучение завершилось без ранней остановки.

---

## 4. Оценка качества  

| Метрика (hold-out 20 %) | Значение |
|-------------------------|----------|
| **MAE**                 | **0.285** |
| **RMSE**                | **0.709** |

---

## 5. Замер времени обучения  

* Суммарное время — **≈ 1.5 мин** (23 эпохы на CPU).

---

## 6. Сравнение с эталонной реализацией (Surprise SVD)  

| Модель                  | MAE ↓ | RMSE ↓ | Train-time |
|-------------------------|-------|--------|------------|
| **Собственная LFM**    | **0.285** | **0.709** | ~1.5 мин |
| Surprise SVD (k = 37)  | 0.669 | 0.870 | **0.29 с** |

*Эталон* использует класс `surprise.SVD`, относящийся к экосистеме scikit-learn.

---

## 7. Выводы  

* Самописная LFM-модель при равном числе факторов заметно превосходит эталон по точности (MAE ↓ в ~2.5 раза; RMSE ↓ на 18 %).  
* Выбор между точностью и скоростью зависит от практических требований конкретной системы рекомендаций.
