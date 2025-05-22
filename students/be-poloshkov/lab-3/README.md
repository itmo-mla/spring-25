# Лабораторная работа №3. Наивный байесовский классификатор

В рамках данной лабораторной работы предстоит реализовать наивный байесовский классификатор и сравнить его с эталонной реализацией из библиотеки `scikit-learn`.

## Задание

1. Выбрать датасет для анализа, например, на [kaggle](https://www.kaggle.com/datasets).
2. Реализовать наивный байесовский классификатор.
3. Обучить модель на выбранном датасете.
4. Оценить качество модели с использованием кросс-валидации.
5. Замерить время обучения модели.
6. Сравнить результаты с эталонной реализацией из библиотеки [scikit-learn](https://scikit-learn.org/stable/):
   * точность модели;
   * время обучения.
7. Подготовить отчет, включающий:
   * описание наивного байесовского классификатора;
   * описание датасета;
   * результаты экспериментов;
   * сравнение с эталонной реализацией;
   * выводы.

### Датасет

Для анализа был выбран датасет [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### Описание наивного байесовского классификатора

Формула Байеса:

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

Где:
- `P(A|B)` — апостериорная вероятность события `A` при условии `B`,
- `P(B|A)` — правдоподобие (вероятность `B` при условии `A`,
- `P(A)` — априорная вероятность `A`,
- `P(B)` — вероятность события `B` (нормирующая константа).

Алгоритм работает следующим образом:

* Вычисляются априорные вероятности `P(C)` для каждого класса.
* Для каждого признака вычисляется вероятность `P(Xi|C)`
* Для нового объекта вычисляется вероятность принадлежности к каждому классу и выбирается класс с максимальной вероятностью.

Априорная вероятность класса `C` вычисляется как

$$
P(C) = \frac{\text{Количество объектов класса } C}{\text{Общее количество объектов}}
$$

Априорная вероятность признака `Xi` с помощью сглаживания Лапласа

$$
P(X_i | C) = \frac{\text{Количество объектов класса } C \text{ с признаком } X_i + \alpha}{\text{Количество объектов класса } C + \alpha \cdot K}
$$
где:
- `a` — параметр сглаживания (обычно `a = 1`),
- `K` — количество возможных значений признака `xi`.

Апостериорная вероятность класса `C` вычисляется как

$$
P(C | X_1, \dots, X_n) = \frac{P(C) \cdot \prod_{i=1}^n P(X_i | C)}{P(X_1, \dots, X_n)}
$$

`P(X)` не зависит от класса, а значит является константой и поэтому его можно убрать из знаменателя и сравнивать только числитель. 

Чтобы избежать численной нестабильности, используется логарифмирование:

$$
\log \big( P(C | X) \big) = \log \big( P(C) \big) + \sum_{i} \log \big( P(X_i | C) \big)
$$

Далее выбирается класс `Ci` с наибольшим значением `log(P(Ci|X))`. 

### Реализация наивного байесовского классификатора

```python
import numpy as np

from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.word_likelihoods = defaultdict(dict)
        self.classes = None
        self.vocab = None

    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_priors = {cls: count / len(y) for cls, count in zip(self.classes, counts)}

        for cls in self.classes:
            cls_X = X[y == cls]
            total_words = 0
            word_counts = defaultdict(int)
            for doc in cls_X:
                words = doc.split()
                total_words += len(words)
                for word in words:
                    word_counts[word] += 1

            for word, count in word_counts.items():
                self.word_likelihoods[cls][word] = (count + 1) / (total_words + len(word_counts))

    def predict(self, X):
        predictions = []
        for doc in X:
            log_probs = {}
            for cls in self.classes:
                log_prob = np.log(self.class_priors[cls])
                for word in doc.split():
                    log_prob += np.log(self.word_likelihoods[cls].get(word, 1e-6))
                log_probs[cls] = log_prob
            predictions.append(max(log_probs, key=log_probs.get))
        return predictions
```

### Сравнение результатов

| Показатель      | Sklearn         | Custom          |
|-----------------|-----------------|-----------------|
| Время работы    | 0.0047          | 0.0114          |
| Accuracy        | 0.9839          | 0.9848          |
| Кросс-валидация | 0.9961 ± 0.0057 | 0.9808 ± 0.0017 |

## Выводы

Различия в точности моделей пренебрежимо малы, а кросс-валидация показывает малое отклонение, что означает стабильность модели.
SKLEARN показал себя в 2.5 раза быстрее, чем собственная реализация
