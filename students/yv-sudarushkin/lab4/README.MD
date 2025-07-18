# Лабораторная работа №4. Латентное размещение Дирихле (LDA)

**Цель:** Реализовать алгоритм латентного размещения Дирихле (LDA) и сравнить его с эталонной реализацией из библиотеки `scikit-learn`.


## Описание алгоритма

**Latent Dirichlet Allocation (LDA)** — это вероятностная модель, которая предполагает, что каждый документ представляет собой смесь скрытых тем, а каждая тема — распределение вероятностей по словам.

Модель обучается методом **Collapsed Gibbs Sampling**:  
- Каждому слову в каждом документе случайно присваивается тема.  
- Далее повторяется пересэмплирование тем для всех слов на основе апостериорных вероятностей.  
- Постепенно формируются устойчивые распределения тем по документам (θ) и слов по темам (φ).

Для сглаживания используются параметры α и β (гиперпараметры распределений Дирихле).


## Описание датасета

В использован датасет:

[60k Stack Overflow Questions](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate)

- Использован подмассив из **2000 случайных вопросов** (`train.csv`)
- Каждое наблюдение состоит из **заголовка** и **текста вопроса**
- Предобработка:
  - Удаление HTML
  - Токенизация, лемматизация, удаление стоп-слов (`nltk`)
  - Векторизация: `CountVectorizer(max_df=0.95, min_df=5, stop_words='english')`


## Результаты экспериментов

| Модель       | Время обучения | Когерентность (c_v) |
|--------------|----------------|----------------------|
| `LDA_custom` | 294.20 сек     | **0.5854**           |
| `sklearn`    | **12.68 сек**      | 0.5367               |

- Количество тем: **7**


## Выводы

- **Собственная реализация LDA** показала чуть выше **когерентность тем** по сравнению с моделью `sklearn`, что указывает на корректную реализацию базового алгоритма.
- Однако, **время обучения** собственной реализации существенно выше, что объясняется отсутствием оптимизаций и матричных ускорений.
