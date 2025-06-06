# Лабораторная работа №4: Реализация Латентного размещения Дирихле (LDA)

## Метрики производительности

### Анализ тем на наборе данных 20 Newsgroups

#### Метрики когерентности тем

| Метрика | Моя реализация | Sklearn |
|---------|---------------|---------|
| UMass | -6.4922 | -7.3528 |
| c_v | 0.4824 | 0.4500 |

#### Время выполнения

| Операция | Моя реализация (сек) | Sklearn (сек) |
|----------|----------------------|---------------|
| Обучение | 108.53 | 90.70 |

## Анализ результатов

- Моя реализация достигла лучшей когерентности тем по обеим метрикам: UMass (-6.49 против -7.35) и c_v (0.48 против 0.45).
- Значение UMass ближе к нулю указывает на лучшую когерентность, а более высокое значение метрики c_v также подтверждает лучшую интерпретируемость тем.
- Время выполнения моей реализации примерно на 20% больше, чем у реализации из Sklearn (108.53 сек против 90.70 сек).

