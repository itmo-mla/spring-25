# Лабораторная работа №5

## Датасет

Для решения задачи был выбран датасет MovieLens 100K ([ссылка](https://grouplens.org/datasets/movielens/100k/)), в котором содержится 100k оценок 1682 фильмов, поставленных 943 пользователями. Оценки представляют собой вещественные числа в диапазоне от 1 до 5.

## Описание алгоритма
Модель латентных векторов (Latent Factor Model, LFM) — это метод коллаборативной фильтрации, применяемый в рекомендательных систем и анализа данных. Он основан на представлении объектов (например, пользователей и объектов пользования) в виде векторов в скрытом (латентном) пространстве признаков.

## Реализация
При обучении модели (см. метод `fit`) для каждого пользователя и фильма считается предсказанный рейтинг путем умножения вектора латентных факторов пользователя на транспонированный вектор скрытых факторов фильма. Затем вычисляется отклонение предикта от реального рейтинга, использующееся далее для корректировки значений скрытых факторов. Значения изменяются путем прибавления или вычитания (в зависимости от знака ошибки) разности произведения ошибки на значения вектора л. факторов фильма/пользователя и члена регуляризации (произведения коэф. регуляризации на сам вектор). Также учитываются средние рейтинги фильмов и пользователей.

```python
def fit(self, data):
        self.global_mean = data["rating"].mean()
        counts_user = np.bincount(data["user_idx"], minlength=self.n_users)
        counts_item = np.bincount(data["item_idx"], minlength=self.n_items)
        
        for idx, row in data.iterrows():
            u = row['user_idx']
            i = row['item_idx']
            r_ui = row['rating']
            self.user_means[u] += r_ui
            self.item_means[i] += r_ui
        
        valid_users = counts_user > 0
        valid_items = counts_item > 0
        self.user_means[valid_users] /= counts_user[valid_users]
        self.item_means[valid_items] /= counts_item[valid_items]
        
        for epoch in range(self.n_epochs):
            for _, row in data.iterrows():
                u = row['user_idx']
                i = row['item_idx']
                r_ui = row['rating']
                pred = self.global_mean + self.user_means[u] + self.item_means[i] + self.user_factors[u].dot(self.item_factors[i].T)
                error = r_ui - pred
                self.user_factors[u] += self.lr * (error * self.item_factors[i] - self.reg_coef * self.user_factors[u])
                self.item_factors[i] += self.lr * (error * self.user_factors[u] - self.reg_coef * self.item_factors[i])
```
## Сравнение с эталоном
### Метрики
RMSE кастомного алгоритма: 0.985 vs 0.941 scikit-surprise.
MAE кастомного алгоритма: 0.765 vs 0.736 у библиотечной реализации.

### Время обучения
Время обучения кастомного алгоритма: 151.865 с vs 0.996 с библиотечного алгоритма. 

## Выводы
Кастомная LFM приблизилась к библиотечной реализации scikit-surprise по значениям метрик, однако обучалась на порядок дольше. Такая значительная разница во времени обучения может быть объяснена тем фактом, что реализация SVD scikit-surprise написана на Cython и куда более оптимизирована. 
