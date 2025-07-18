# Отчет по лабораторной работе

## Описание алгоритма градиентного бустинга

**Градиентный бустинг** — это ансамблевый метод машинного обучения, который последовательно строит модели для исправления ошибок предыдущих моделей.

Принцип работы:
- Инициализация: начальное предсказание = среднее значение целевой переменной
- Итеративный процесс:
1. Вычисление остатков (градиентов функции потерь)
2. Обучение слабого ученика (дерево решений) на остатках
3. Добавление предсказаний с коэффициентом обучения (learning rate)
4. Финальное предсказание: Сумма всех слабых учеников

## Описание датасета
Датасет недвижимости содержит информацию о 500 объектах недвижимости.

Характеристики:
- Размер: 500 объектов × 11 признаков
- Тип задачи: регрессия (предсказание цены)
- Целевая переменная: Price (от ~277k до ~960k)

## Результаты экспериментов

Параметры модели:
- n_estimators = 50
- learning_rate = 0.1
- max_depth = 4
- random_state = 42

### Итоговые метрики:
- R^2: 0.8915 ± 0.0162
- Время на фолд: 0.10 ± 0.01 секунд
- Общее время: 0.5 секунд

Sklearn реализация:
- R^2: 0.8799 ± 0.0099
- Общее время: 0.41 секунд

## Выводы
- Разность в точности между самописаной моделью и готовой реализацией мала, как и время обучения
- Алгоритм полезен для различных задач, например, регрессионной задачи предсказания цены