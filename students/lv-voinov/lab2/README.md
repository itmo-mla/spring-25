# Лабораторная работа 2

## Датасет

Breast Cancer

<https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset>

Датасет для бинарной классификации. 30 числовых признаков, доля класса 1 63%. Качественный датасет, содержащий малое количество шума.

## Алгоритм градиентного бустинга

Градиентный бустинг — это алгоритм машинного обучения, который строит ансамбль моделей последовательно, корректируя ошибки предыдущих шагов. В данном случае в качестве базовых алгоритмов используются решающие пни - решающие деревья глубиной 1.

Алгоритм начинается с инициализации начального предсказания, например, среднего значения целевой переменной для задач регрессии или начального приближения через логарифмические отношения для классификации. Затем на каждой итерации создаётся новая модель, которая обучается предсказывать не целевую переменную, а градиенты функции потерь, вычисленные относительно текущего ансамбля.

Эти градиенты отражают направление, в котором нужно скорректировать предсказания, чтобы уменьшить ошибку. Для квадратичной потери градиент — это разница между истинным значением и текущим предсказанием, а в общем случае он определяется через производную функции потерь.

Каждая новая модель, часто реализуемая как дерево решений, обучается приближать эти градиенты, минимизируя остаточные ошибки. После обучения модели её предсказания умножаются на коэффициент обучения — малый параметр, который контролирует вклад каждой отдельной модели и предотвращает переобучение.

Обновлённый ансамбль формируется добавлением взвешенного предсказания новой модели к текущему ансамблю. Процесс повторяется заданное количество итераций или до достижения сходимости.

Итоговое предсказание получается суммированием начального приближения и всех скорректированных предсказаний моделей, что позволяет постепенно уменьшать ошибку и адаптироваться к сложным закономерностям в данных. В отличие от методов вроде случайного леса, где модели обучаются независимо, градиентный бустинг явно оптимизирует ансамбль, фокусируясь на ошибках, которые остаются после предыдущих итераций, что делает его мощным инструментом для задач с нелинейными зависимостями и шумом.

### Код алгоритма

```python
class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.initial_pred = None

    def fit(self, X, y):
        pos = np.mean(y)
        epsilon = 1e-10
        
        if pos < epsilon:
            self.initial_pred = -1e10
        elif pos > 1 - epsilon:
            self.initial_pred = 1e10
        else:
            self.initial_pred = np.log(pos / (1 - pos))
        
        F = np.full(X.shape[0], self.initial_pred)
        
        for _ in range(self.n_estimators):
            p = 1 / (1 + np.exp(-F))
            
            residuals = y - p
            
            model = DecisionTreeRegressor(max_depth=1)
            model.fit(X, residuals)
            predictions = model.predict(X)
            
            F += self.learning_rate * predictions
            self.models.append(model)

    def predict(self, X):
        F = np.full(X.shape[0], self.initial_pred)
        
        for model in self.models:
            F += self.learning_rate * model.predict(X)
        
        proba = 1 / (1 + np.exp(-F))
        return proba >= 0.5
```

### Результаты работы ручного алгоритма

Результаты и время работы, посчитанные с помощью кросс-валидации для различных n_estimators и learning_rate:

```
Mean accuracy for n_estimators=50, learning_rate=0.1: 0.9398 (+-0.0117), time: 0:00:00.624805
Mean accuracy for n_estimators=50, learning_rate=0.5: 0.9593 (+-0.0206), time: 0:00:00.571883
Mean accuracy for n_estimators=50, learning_rate=2: 0.9752 (+-0.0142), time: 0:00:00.602605
Mean accuracy for n_estimators=50, learning_rate=5: 0.9823 (+-0.0097), time: 0:00:00.617291
Mean accuracy for n_estimators=100, learning_rate=0.1: 0.9398 (+-0.0117), time: 0:00:01.157570
Mean accuracy for n_estimators=100, learning_rate=0.5: 0.9699 (+-0.0106), time: 0:00:01.138680
Mean accuracy for n_estimators=100, learning_rate=2: 0.9805 (+-0.0180), time: 0:00:01.197076
Mean accuracy for n_estimators=100, learning_rate=5: 0.9841 (+-0.0152), time: 0:00:01.112291
Mean accuracy for n_estimators=200, learning_rate=0.1: 0.9575 (+-0.0205), time: 0:00:02.316778
Mean accuracy for n_estimators=200, learning_rate=0.5: 0.9770 (+-0.0164), time: 0:00:02.350584
Mean accuracy for n_estimators=200, learning_rate=2: 0.9841 (+-0.0197), time: 0:00:02.401747
Mean accuracy for n_estimators=200, learning_rate=5: 0.9858 (+-0.0154), time: 0:00:02.279050
Mean accuracy for n_estimators=500, learning_rate=0.1: 0.9717 (+-0.0103), time: 0:00:05.765404
Mean accuracy for n_estimators=500, learning_rate=0.5: 0.9841 (+-0.0162), time: 0:00:05.612308
Mean accuracy for n_estimators=500, learning_rate=2: 0.9876 (+-0.0164), time: 0:00:05.735124
Mean accuracy for n_estimators=500, learning_rate=5: 0.9894 (+-0.0130), time: 0:00:05.804754
Mean accuracy for n_estimators=1000, learning_rate=0.1: 0.9770 (+-0.0164), time: 0:00:12.038930
Mean accuracy for n_estimators=1000, learning_rate=0.5: 0.9858 (+-0.0199), time: 0:00:11.430698
Mean accuracy for n_estimators=1000, learning_rate=2: 0.9894 (+-0.0172), time: 0:00:11.313317
Mean accuracy for n_estimators=1000, learning_rate=5: 0.9912 (+-0.0137), time: 0:00:11.651124
```

### Результаты работы библиотечного алгоритма

Результаты и время работы, посчитанные с помощью кросс-валидации для различных n_estimators:

```
Mean accuracy for n_estimators=50, learning_rate=0.1: 0.9490 (+-0.0179), time: 0:00:00.564475
Mean accuracy for n_estimators=50, learning_rate=0.5: 0.9649 (+-0.0175), time: 0:00:00.579181
Mean accuracy for n_estimators=50, learning_rate=2: 0.7665 (+-0.2056), time: 0:00:00.572921
Mean accuracy for n_estimators=50, learning_rate=5: 0.3726 (+-0.0039), time: 0:00:00.561660
Mean accuracy for n_estimators=100, learning_rate=0.1: 0.9649 (+-0.0166), time: 0:00:01.147919
Mean accuracy for n_estimators=100, learning_rate=0.5: 0.9631 (+-0.0195), time: 0:00:01.100490
Mean accuracy for n_estimators=100, learning_rate=2: 0.7665 (+-0.2056), time: 0:00:01.132785
Mean accuracy for n_estimators=100, learning_rate=5: 0.3726 (+-0.0039), time: 0:00:01.145645
Mean accuracy for n_estimators=200, learning_rate=0.1: 0.9666 (+-0.0140), time: 0:00:02.167629
Mean accuracy for n_estimators=200, learning_rate=0.5: 0.9701 (+-0.0197), time: 0:00:02.231389
Mean accuracy for n_estimators=200, learning_rate=2: 0.7665 (+-0.2056), time: 0:00:02.239833
Mean accuracy for n_estimators=200, learning_rate=5: 0.3726 (+-0.0039), time: 0:00:02.348559
Mean accuracy for n_estimators=500, learning_rate=0.1: 0.9719 (+-0.0151), time: 0:00:05.424176
Mean accuracy for n_estimators=500, learning_rate=0.5: 0.9719 (+-0.0195), time: 0:00:05.667532
Mean accuracy for n_estimators=500, learning_rate=2: 0.7665 (+-0.2056), time: 0:00:05.454601
Mean accuracy for n_estimators=500, learning_rate=5: 0.3726 (+-0.0039), time: 0:00:05.524915
Mean accuracy for n_estimators=1000, learning_rate=0.1: 0.9684 (+-0.0212), time: 0:00:11.003143
Mean accuracy for n_estimators=1000, learning_rate=0.5: 0.9737 (+-0.0200), time: 0:00:08.766117
Mean accuracy for n_estimators=1000, learning_rate=2: 0.7665 (+-0.2056), time: 0:00:11.143264
Mean accuracy for n_estimators=1000, learning_rate=5: 0.3726 (+-0.0039), time: 0:00:11.050472
```

### Выводы

В результате экспериментов было выяснено, что оптимальный learning rate для ручного алгоритма очень большой и равен 5. Лучшая точность в этом случае достигается при использовании 1000 алгоритмов: 0.9912. У библиотечного алгоритма при этом оптимальный learning rate - 0.5, а при слишком большом значении алгоритм перестает работать. Точность при том же количестве алгоритмов получается намного меньше: всего 0.9737, что на 0.017 ниже, чем у ручного.

Время работы ручного и библиотечного алгоритмов примерно равно, но на большом количестве классификаторов библиотечный алгоритм становится чуть быстрее.

В целом, реализованный алгоритм является более эффективным для классификации на выбранном датасете, чем алгоритм из библиотеки.
