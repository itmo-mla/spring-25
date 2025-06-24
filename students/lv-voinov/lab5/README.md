# Лабораторная работа 5

## Датасет

MovieLens 100K

<https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset>

Датасет с 100000 оценок фильмов от 1 до 5 от 943 пользователей.

## Модель латентных факторов

Модель латентных факторов — ключевой подход в рекомендательных системах, основанный на выявлении скрытых характеристик пользователей и объектов. Она декомпозирует матрицу взаимодействий (например, рейтинги фильмов) в произведение двух матриц меньшей размерности: пользовательских и предметных векторов в общем латентном пространстве. Предсказание формируется как скалярное произведение этих векторов с добавлением базовых смещений — общего среднего, индивидуальных предубеждений пользователей и популярности объектов.

Обучение модели минимизирует функцию потерь, сочетающую ошибки предсказаний и регуляризационные слагаемые. Через стохастический градиентный спуск итеративно корректируются параметры: на каждом шаге векторы пользователя и объекта обновляются пропорционально ошибке предсказания для конкретного взаимодействия. Критически важны гиперпараметры: количество факторов (определяет сложность модели), скорость обучения (влияет на сходимость) и регуляризация (контролирует переобучение).

Главные преимущества включают способность выявлять сложные паттерны в разреженных данных и хорошую масштабируемость. Однако эффективность реализации сильно зависит от оптимизации вычислений — векторизованные операции в специализированных библиотеках (например Surprise) обеспечивают на порядки большее быстродействие по сравнению с наивной реализацией на Python. Эта модель остается теоретическим фундаментом современных рекомендательных систем, несмотря на появление более сложных архитектур.

### Код алгоритма

```python
class LatentFactorModel:
    def __init__(self, n_factors=50, learning_rate=0.005, reg=0.02, n_epochs=50):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        
    def fit(self, train):
        self.global_mean = train.rating.mean()
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.user_factors = np.random.normal(scale=1/self.n_factors, size=(n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=1/self.n_factors, size=(n_items, self.n_factors))
        
        for epoch in range(self.n_epochs):
            for user, item, rating in train[['user_id', 'item_id', 'rating']].values:
                user, item = int(user)-1, int(item)-1
                
                prediction = (
                    self.global_mean 
                    + self.user_biases[user] 
                    + self.item_biases[item] 
                    + np.dot(self.user_factors[user], self.item_factors[item])
                )
                
                error = rating - prediction
                
                self.user_biases[user] += self.lr * (error - self.reg * self.user_biases[user])
                self.item_biases[item] += self.lr * (error - self.reg * self.item_biases[item])
                
                uf = self.user_factors[user]
                itf = self.item_factors[item]
                
                self.user_factors[user] += self.lr * (error * itf - self.reg * uf)
                self.item_factors[item] += self.lr * (error * uf - self.reg * itf)
    
    def predict(self, test):
        preds = []
        for user, item in test[['user_id', 'item_id']].values:
            user, item = int(user)-1, int(item)-1
            pred = (
                self.global_mean 
                + self.user_biases[user] 
                + self.item_biases[item] 
                + np.dot(self.user_factors[user], self.item_factors[item])
            )
            preds.append(pred)
        return np.clip(preds, 1, 5)
```

### Результаты работы

Было проведено сравнение результатов работы ручного алгоритма и алгоритма из библиотеки Surprise.

Результаты ручного алгоритма:

```
Custom Model: RMSE = 0.9156, MAE = 0.7173, Time = 101.58s
```

Результаты библиотечного алгоритма:

```
Surprise Model: RMSE = 0.9714, MAE = 0.7590, Time = 2.12s
```

### Выводы

Метрики отклонений RMSE и MAE у ручной реализации лучше, но библиотечная реализация работает в 50 раз быстрее.

## Задание

Взята История игрушек и еще 2 фильма Pixar (id: 1, 71, 993).

Код для рекомендации:

```python
movies = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.item', 
    sep='|', 
    encoding='latin-1',
    header=None,
    names=['item_id', 'title'] + [f'f{i}' for i in range(23)]
)[['item_id', 'title']]

all_movies = ratings['item_id'].unique()

input_data = pd.DataFrame({
    'user_id': [new_user_id] * len(all_movies),
    'item_id': all_movies
})

predictions = model.predict(input_data)

results = input_data.copy()
results['predicted_rating'] = predictions

results = results.merge(movies, on='item_id', how='left')

top_10 = results.sort_values('predicted_rating', ascending=False).head(10)
for i, row in enumerate(top_10.itertuples(), 1):
    print(f"{i}. {row.title} [{row.item_id}]")
```

Результаты рекомендации:

```
1. Casablanca (1942) [483]
2. Schindler's List (1993) [318]
3. Wrong Trousers, The (1993) [169]
4. Wallace & Gromit: The Best of Aardman Animation (1996) [114]
5. Usual Suspects, The (1995) [12]
6. Shawshank Redemption, The (1994) [64]
7. Rear Window (1954) [603]
8. Pather Panchali (1955) [1449]
9. 12 Angry Men (1957) [178]
10. Star Wars (1977) [50]
```
