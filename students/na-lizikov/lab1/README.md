# Лабораторная работа №1. Ансамбли моделей

## Описание метода

Bagging (Bootstrap Aggregating) - это ансамблевый метод машинного обучения, который комбинирует предсказания нескольких базовых моделей для улучшения общей производительности и уменьшения переобучения. Основные шаги метода:

1. Создание бутстрап-выборок из исходного набора данных
2. Обучение базового алгоритма на каждой выборке
3. Агрегация предсказаний всех моделей (голосование большинством для классификации)

## Описание датасета

В данной работе используется датасет Breast Cancer Wisconsin в формате CSV. Датасет содержит характеристики клеточных ядер, полученных из цифровых изображений тонкоигольной аспирационной биопсии молочной железы.

### Структура данных:
- id: идентификационный номер образца
- diagnosis: диагноз (M = злокачественная опухоль, B = доброкачественная опухоль)
- 30 признаков, описывающих характеристики ядер клеток:
  - radius (средний радиус)
  - texture (среднее отклонение значений серого)
  - perimeter (средний периметр)
  - area (средняя площадь)
  - smoothness (средняя гладкость)
  - и другие характеристики

### Характеристики датасета:
- Количество признаков: 30
- Количество классов: 2 (злокачественная/доброкачественная опухоль)
- Размер выборки: 569 образцов
- Формат: CSV файл

## Экспериментальные результаты

Сравнение производительности ручной реализации и реализации из scikit-learn проводилось по следующим параметрам:
- Точность классификации (cross-validation)
- Время обучения

| Реализация | Средняя точность | Стандартное отклонение | Время обучения (сек) |
|------------|------------------|------------------------|----------------------|
| Ручная     | 0.956125	        | 0.0221                 | 0.5627               |
| Sklearn    | 0.949092         | 0.0290                 | 0.3819               |

## Выводы

1. Обе реализации показывают схожую точность классификации
2. Реализация scikit-learn работает быстрее за счет оптимизации и использования параллельных вычислений
