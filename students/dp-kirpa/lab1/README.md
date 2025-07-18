# Лабораторная работа №1. Ансамбли моделей

Был использован набор данных https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset. В нём описываются параметры тренировок в спортивном зале.

Я провёл 2 эксперимента по сравнению ручного бустинга и библиотечной версии. В качестве целевой переменной для первого эксперимента я использовал тип тренировки (силовая, аэробная и тд). В качестве целевой переменной для второго эксперимента я использовал пол тренирующегося.

Для каждого эксперимента я снимал метрики моей реализации бустинга, библиотечной версии (`AdaBoostClassifier`) и библиотечного дерева (`DecisionTreeClassifier`).

Для первого эксперимента дерево работало чуть лучше монетки (accuracy = 0.518). Но ансамбль из библиотеки, получив 100%-ную точность на обучающем датасете, на тесте дал 0.50, то есть ровно монетку. Моя реализация при глубине в одно дерево дала 0.482 (то есть ровно базовое решающее дерево, но с инвертированными метками классов). При использовании большего кол-ва базовых деревьев результат стал 0.518, то есть ровно базовое решающее дерево. Базовое дерево переобучается на исходных данных и его использование в ансамбле качеству не помогло.

Для второго эксперимента дерево дало 0.8 accuracy. Оба ансамбля дали одинаковое accuracy = 0.995.

Из экспериментов делаю вывод о том, что моя реализация рабочая и библиотечной версии по качеству не уступает. Скорость обучения сопоставима (оба учатся порядка секунды).