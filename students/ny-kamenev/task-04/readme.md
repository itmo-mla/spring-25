# Латентное размещение Дирихле (LDA)

В данной лабораторной работе реализован алгоритм латентного размещения Дирихле и выполнено сравнение с эталонной реализацией из библиотеки sklearn.

## Датасеты

Для анализа были выбраны следующие датасеты:

1. **20 Newsgroups** - коллекция новостных сообщений, разделенных на 20 различных категорий
2. **BBC News** - коллекция новостных статей BBC, разделенных на 5 категорий
3. **Reuters-21578** - коллекция новостных сообщений Reuters, разделенных на 8 основных тем

## Реализация

Реализован алгоритм LDA с использованием:
- EM-алгоритма для обучения
- Метрик когерентности тем (UMass и C_v) для оценки качества
- Оптимизации гиперпараметров (alpha=0.1, beta=0.01)

### Результаты

#### 20 Newsgroups Dataset (20 тем)

| Метод | Время обучения | UMass | C_v |
|-------|----------------|-------|-----|
| Custom LDA | 305.84s | -6.0213 | 0.4834 |
| Sklearn LDA | 253.60s | -6.4149 | 0.4606 |

Топ-5 слов для каждой темы:
- Topic 1: cx, w7, ah, lk, chz
- Topic 2: know, don, just, think, like
- Topic 3: drive, windows, card, scsi, use
- Topic 4: don, people, like, just, think
- Topic 5: god, jesus, does, people, believe
- Topic 6: key, encryption, chip, government, keys
- Topic 7: president, gun, government, people, mr
- Topic 8: edu, graphics, image, fax, available
- Topic 9: edu, com, mail, information, file
- Topic 10: 00, new, price, 50, sale
- Topic 11: health, use, medical, new, disease
- Topic 12: game, team, year, games, season
- Topic 13: window, use, program, server, file
- Topic 14: ax, max, g9v, b8f, a86
- Topic 15: space, nasa, launch, earth, data
- Topic 16: hi, gm, john, appreciated, st
- Topic 17: armenian, israel, armenians, turkish, jews
- Topic 18: car, db, like, bike, cars
- Topic 19: output, file, entry, line, program
- Topic 20: 10, 12, 11, 15, 14

#### BBC News Dataset (5 тем)

| Метод | Время обучения | UMass | C_v |
|-------|----------------|-------|-----|
| Custom LDA | 136.48s | -2.8993 | 0.4612 |
| Sklearn LDA | 29.93s | -1.7849 | 0.5516 |

Топ-5 слов для каждой темы:
- Topic 1: film, best, company, new, sales
- Topic 2: mr, government, labour, election, party
- Topic 3: people, technology, music, mobile, new
- Topic 4: game, time, england, win, world
- Topic 5: mr, people, told, law, new

#### Reuters Dataset (8 тем)

| Метод | Время обучения | UMass | C_v |
|-------|----------------|-------|-----|
| Custom LDA | 0.05s | -4.7068 | 0.3811 |
| Sklearn LDA | 0.12s | -7.1084 | 0.3715 |

Топ-5 слов для каждой темы:
- Topic 1: dlrs, new, sales, mln, year
- Topic 2: 000, mln, year, net, sales
- Topic 3: sales, pct, year, earlier, corp
- Topic 4: stock, dlrs, shares, earlier, year
- Topic 5: company, pct, corp, earlier, dlrs
- Topic 6: mln, pct, net, earlier, year
- Topic 7: pct, new, sales, shares, stock
- Topic 8: mln, corp, dlrs, year, pct

## Выводы

1. **Производительность**:
   - На больших датасетах (20 Newsgroups) sklearn LDA работает быстрее (253.60s vs 305.84s)
   - На средних датасетах (BBC) разница в производительности существенна (29.93s vs 136.48s)
   - На малых датасетах (Reuters) оба метода работают очень быстро

2. **Метрики**:
   - Метрика UMass показывает лучшие результаты для sklearn LDA на BBC датасете (-1.7849 vs -2.8993)
   - Метрика C_v показывает лучшие результаты для custom LDA на 20 Newsgroups (0.4834 vs 0.4606)
   - На Reuters датасете custom LDA показывает лучшие результаты по обеим метрикам
