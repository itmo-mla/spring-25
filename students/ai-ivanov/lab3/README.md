# Наивный байесовский классификатор

## Набор данных

https://www.kaggle.com/datasets/abrambeyer/openintro-possum

Набор данных по опоссумам состоит из девяти морфометрических измерений, сделанных на 104 горных щеткохвостых опоссумах, пойманных в семи местах от Южной Виктории до центрального Квинсленда.

| site | Pop | sex | age | hdlngth | skullw | totlngth | taill | footlgth | earconch | eye | chest | belly |
|------|-----|-----|-----|---------|---------|-----------|--------|-----------|-----------|-----|--------|--------|
| 1 | Vic | m | 8.0 | 94.1 | 60.4 | 89.0 | 36.0 | 74.5 | 54.5 | 15.2 | 28.0 | 36.0 |
| 1 | Vic | f | 6.0 | 92.5 | 57.6 | 91.5 | 36.5 | 72.5 | 51.2 | 16.0 | 28.5 | 33.0 |
| 1 | Vic | f | 6.0 | 94.0 | 60.0 | 95.5 | 39.0 | 75.4 | 51.9 | 15.5 | 30.0 | 34.0 |
| 1 | Vic | f | 6.0 | 93.2 | 57.1 | 92.0 | 38.0 | 76.1 | 52.2 | 15.2 | 28.0 | 34.0 |
| 1 | Vic | f | 2.0 | 91.5 | 56.3 | 85.5 | 36.0 | 71.0 | 53.2 | 15.1 | 28.5 | 33.0 |

## Теоретическая часть

Наивный байесовский классификатор - это вероятностный классификатор, основанный на применении теоремы Байеса с предположением о независимости признаков.

### Теорема Байеса

В основе классификатора лежит теорема Байеса:

$ P(y|x) = \frac{P(x|y)P(y)}{P(x)} $

где:
- $P(y|x)$ - апостериорная вероятность класса $y$ при наблюдении признака $x$
- $P(x|y)$ - правдоподобие
- $P(y)$ - априорная вероятность класса
- $P(x)$ - вероятность признака

### Принцип работы классификатора

Для набора признаков $X = (x_1, ..., x_n)$ классификатор определяет наиболее вероятный класс $y$:

$ y = \arg\max_{y} P(y|X) = \arg\max_{y} P(y)\prod_{i=1}^{n} P(x_i|y) $

Предположение о независимости признаков позволяет нам записать:

$ P(X|y) = \prod_{i=1}^{n} P(x_i|y) $

### Типы наивных байесовских классификаторов

1. **Гауссовский наивный Байес**
   - Предполагает нормальное распределение числовых признаков
   - $ P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i-\mu_y)^2}{2\sigma_y^2}\right) $

2. **Мультиномиальный наивный Байес**
   - Используется для дискретных признаков (например, частоты слов в тексте)
   - $ P(x_i|y) = \frac{\text{count}(x_i,y) + \alpha}{\text{count}(y) + \alpha n} $

3. **Бернуллиевский наивный Байес**
   - Для бинарных признаков
   - $ P(x_i|y) = P(i|y)^{x_i} (1-P(i|y))^{(1-x_i)} $

### Методы сглаживания вероятностей

При работе с наивным байесовским классификатором часто возникает проблема "нулевой вероятности" - когда некоторые комбинации признак-класс не встречаются в обучающих данных. Для решения этой проблемы применяются различные методы сглаживания:

1. **Сглаживание Лапласа (аддитивное сглаживание)**
   - Добавляет псевдо-счётчик α к каждому наблюдению
   - $ P(x_i|y) = \frac{count(x_i,y) + \alpha}{count(y) + \alpha N} $
   - Значение α = 1.0 (классическое сглаживание Лапласа)

2. **Сглаживание дисперсии**
   - Применяется для гауссовского распределения
   - Добавляет малую константу к дисперсии: $ \sigma^2_{smooth} = \sigma^2 + \epsilon $
   - Предотвращает проблемы с нулевой дисперсией

3. **Сглаживание априорных вероятностей**
   - Небольшое сглаживание (α = 1e-10) для вероятностей классов
   - $ P(y) = \frac{count(y) + \alpha}{N + \alpha K} $
   - Где K - количество классов

Применение сглаживания делает классификатор более устойчивым и помогает избежать проблем с численной стабильностью при вычислениях.


### Преимущества и недостатки

**Преимущества:**
- Простота реализации
- Быстрое обучение и классификация
- Хорошая работа с высокоразмерными данными
- Эффективность при небольшом наборе обучающих данных

**Недостатки:**
- Предположение о независимости признаков (часто нереалистично)
- Чувствительность к нерелевантным признакам
- Проблема "нулевой вероятности"

## Реализация на python

```python
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
```

## Экспериментальные результаты

### Описание эксперимента

Для тестирования реализованного наивного байесовского классификатора был использован набор данных о поссумах (possum dataset), где решалась задача классификации пола животного по различным морфометрическим признакам. Данные были разделены на обучающую (80%) и тестовую (20%) выборки.

### Результаты классификации

Self-made реализация показала следующие результаты:

```
              precision    recall  f1-score   support

      female       0.70      0.78      0.74         9
        male       0.82      0.75      0.78        12

    accuracy                           0.76        21
   macro avg       0.76      0.76      0.76        21
weighted avg       0.77      0.76      0.76        21
```

### Сравнение с эталонной реализацией (sklearn)

Реализация из библиотеки sklearn показала схожие результаты:

```
              precision    recall  f1-score   support

      female       0.64      0.78      0.70         9
        male       0.80      0.67      0.73        12

    accuracy                           0.71        21
   macro avg       0.72      0.72      0.71        21
weighted avg       0.73      0.71      0.72        21
```

#### Время обучения

- Self-made реализация: `0.0005` сек
- Sklearn реализация: `0.0008` сек
- Self-made быстрее в `1.5` раза

### Выводы

1. Реализованный классификатор показывает результаты, сопоставимые с эталонной реализацией из библиотеки sklearn, что подтверждает корректность реализации.
2. Self-made реализация даже немного превосходит sklearn по точности (76% vs 71%), что может быть связано с особенностями реализации сглаживания вероятностей и численной стабильности вычислений.
3. Вероятности, предсказанные обеими реализациями, очень близки, что дополнительно подтверждает правильность работы классификатора.
4. Классификатор показывает хорошую способность различать классы, с небольшим преимуществом в определении особей мужского пола (f1-score 0.78 vs 0.74 для женского).
