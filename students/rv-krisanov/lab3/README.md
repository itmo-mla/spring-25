# Наивный байесовский классификатор



## Набор данных


https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

# Исследование наивного байесовского классификатора

## Описание наивного байесовского классификатора

Наивный байесовский классификатор - это вероятностный классификатор, основанный на теореме Байеса с предположением о независимости признаков. Несмотря на это "наивное" предположение, классификатор часто показывает хорошие результаты на практике.

Основная формула классификатора:
$$ P(y|x) = \frac{P(x|y)P(y)}{P(x)} $$

где:
- $P(y|x)$ - апостериорная вероятность класса y при заданных признаках x
- $P(x|y)$ - правдоподобие (likelihood) признаков x для класса y
- $P(y)$ - априорная вероятность класса y
- $P(x)$ - нормализующая константа

В случае независимости признаков (наивное предположение):
$$ P(x|y) = \prod_{i=1}^n P(x_i|y) $$

Для непрерывных признаков используется гауссовское распределение:
$$ P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}) $$

где:
- $\mu_y$ - среднее значение признака для класса y
- $\sigma_y$ - стандартное отклонение признака для класса y

Основные преимущества:
- Простота реализации
- Быстрое обучение
- Хорошая работа с многомерными данными
- Эффективность при малом количестве обучающих данных
- Хорошо работает с несбалансированными данными

## Описание датасета

Использовался датасет [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) с Kaggle. Датасет содержит информацию о различных астрономических объектах, собранную в рамках проекта Sloan Digital Sky Survey (SDSS).

Основные признаки:
- u, g, r, i, z: значения яркости в разных спектральных диапазонах
- redshift: красное смещение
- alpha, delta: координаты объекта
- class: класс объекта (GALAXY, QSO, STAR)

## Результаты экспериментов

### Предобработка данных

В процессе предобработки данных были выполнены следующие шаги:
1. Удаление неинформативных признаков (ID, координаты и т.д.)
2. Стандартизация числовых признаков (u, g, r, i, z, redshift) с помощью StandardScaler
3. Разделение данных на обучающую и тестовую выборки в соотношении 80:20

### Реализация классификатора

Была реализована собственная версия наивного байесовского классификатора с поддержкой:
- Гауссовского распределения для непрерывных признаков
- Мультиномиального распределения для дискретных признаков
- Сглаживания Лапласа для обработки неизвестных значений

### Оценка качества

Для оценки качества модели использовалась кросс-валидация с 5 фолдами. Результаты:

| Какая часть датасета тестовая | Собственная реализация | Библиотечная реализация |
|-----------------------------|----------------------|------------------------|
| 0-20%                       | 0.898511850703634    | 0.898511850703634      |
| 20-40%                      | 0.9141204113501427   | 0.9141204113501427     |
| 40-60%                      | 0.9034207415490236   | 0.9034207415490236     |
| 60-80%                      | 0.9142419517040008   | 0.9142419517040008     |
| 80-100%                     | 0.9082960850279123   | 0.9082960850279123     |


Среднее время обучения и предсказания:  4.4 ms ± 62.9 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

### Сравнение с scikit-learn

Сравнение с реализацией из scikit-learn:

| Метрика | Собственная реализация | Библиотечная реализация |
|---------|-----------------|--------------|
| Время выполнения | 4.4 ms ± 62.9 μs | 4.47 ms ± 41 μs |

## Выводы

1. Наша реализация показала результаты, идентичные эталонной реализации scikit-learn
2. Время выполнения практически идентично (разница менее 1%), что говорит о хорошей оптимизации нашей реализации


В целом, результаты подтверждают эффективность наивного байесовского классификатора для задачи классификации астрономических объектов, особенно учитывая слабую зависимость между признаками в данном датасете.
