# Лабораторная работа №2: Градиентный бустинг

## Описание проекта

В рамках данной лабораторной работы реализован алгоритм градиентного бустинга для задачи регрессии и проведено его сравнение с эталонной реализацией `GradientBoostingRegressor` из библиотеки scikit-learn.

## Описание алгоритма Gradient Boosting

1. **Инициализация**: стартовое предсказание берётся как среднее значение таргета:
   $F_0(x) = \bar y$.
2. **Итерации** (для $m=1\dots M$):

   * Вычисляем остатки (градиенты) для MSE: $r_i = y_i - F_{m-1}(x_i)$.
   * Обучаем дерево решений на признаках $X$ и целевых $r$.
   * Обновляем предсказание: $F_m(x) = F_{m-1}(x) + \eta \cdot tree_m(x)$, где $\eta$ — скорость обучения.
3. **Предсказание**: суммируем начальную константу и вклады всех деревьев.

Параметры:

* `learning_rate` (скорость обучения)
* `n_estimators` (число деревьев)
* `max_depth` (максимальная глубина дерева)

## Датасет

Используется датасет «Bike Sharing Demand» с Kaggle:

* `datetime` — временная метка
* `season`, `holiday`, `workingday`, `weather` — категориальные признаки
* `temp`, `atemp`, `humidity`, `windspeed` — числовые признаки
* `casual`, `registered` — вспомогательные столбцы (убраны из обучения)
* `count` — целевая переменная (количество аренд)

В предобработке:

* Удалены поля `datetime`, `casual`, `registered`.
* Пропуски отсутствуют.

## Эксперименты и результаты

Параметры моделей:

* `learning_rate = 0.1`
* `n_estimators = 100`
* `max_depth = 3`

| Модель                                | MAE (mean ± std) | MSE (mean ± std)  | Время, с |
| ------------------------------------- | ---------------- | ----------------- | -------- |
| Custom `GBMRegressor`                 | 109.27 ± 2.10    | 22018.26 ± 843.71 | 11.18    |
| `GradientBoostingRegressor` (sklearn) | 109.26 ± 2.08    | 22016.67 ± 841.07 | 10.56    |

## Сравнение и выводы
Как видно, качество предсказаний (MAE и MSE) у собственной реализации практически не отличается от библиотеки sklearn, что подтверждает корректность алгоритма. В то же время эталонная модель обучается быстрее: \~10.5 с против \~11.2 с у ручного кода.

* **Качество** (MAE/MSE) ручной реализации практически совпадает с эталонной.
* **Скорость**: реализация из scikit-learn выигрывает \~10–15% по времени обучения.

## Правила использования

* Для изменения гиперпараметров достаточно передать другие значения в конструктор `GBMRegressor` или `GradientBoostingRegressor`.
* Дальнейшие эксперименты можно проводить в `lab2.ipynb`.
