# Лабораторная работа №4: Latent Dirichlet Allocation (LDA)

## Цель работы
Реализация алгоритма LDA (Latent Dirichlet Allocation) для тематического моделирования текстовых данных и сравнение результатов с референсной реализацией из библиотеки scikit-learn.

## Используемый датасет
- Набор новостных статей (2225 документов)
- Категории: business, entertainment, tech, politics
- Структура данных: категория, имя файла, заголовок, содержание

## Алгоритм LDA

### Теоретические основы
LDA (Latent Dirichlet Allocation) - это вероятностная модель, которая позволяет обнаруживать скрытые темы в коллекции документов. Алгоритм основан на следующих предположениях:
1. Каждый документ представляет собой смесь тем
2. Каждая тема представляет собой распределение слов
3. Слова в документе генерируются через процесс:
   - Выбор темы из распределения тем документа
   - Выбор слова из распределения слов выбранной темы

### Математическая модель
- α (alpha) - параметр априорного распределения Дирихле для распределения тем в документах
- β (beta) - параметр априорного распределения Дирихле для распределения слов в темах
- θ - распределение тем в документе
- φ - распределение слов в теме
- z - скрытая тема
- w - наблюдаемое слово

### Процесс обучения
1. Инициализация:
   - Случайное присвоение тем словам
   - Подсчет статистик документ-тема и тема-слово

2. Итеративное обновление:
   - Для каждого документа и слова:
     - Исключение текущего слова из статистик
     - Пересчет вероятностей тем
     - Выбор новой темы
     - Обновление статистик

### Реализация алгоритма

```python
import numpy as np


def customLDA(
    doc_term_matrix: np.ndarray,  # [document_index, word_index] = word_count
    topic_count: int = 5,
    alpha: float = 1,
    beta: float = 0.1,
    iteration: int = 50,
):
    doc_term_matrix = doc_term_matrix.astype(np.int32)
    [D, V], T = (
        doc_term_matrix.shape,
        topic_count,
    )  # documents, vocabulary size, topic count

    phi = np.zeros((T, V), dtype=np.int32)  # темы × слова
    theta = np.zeros((D, T), dtype=np.int32)  # документы × темы

    rows, cols = np.where(doc_term_matrix > 0)
    counts = doc_term_matrix[rows, cols]

    token_document_map = np.repeat(rows, counts)  # (N_tokens,)
    token_word_map = np.repeat(cols, counts)
    token_topic_map = np.random.randint(
        0, topic_count, token_document_map.size, dtype=np.int32
    )

    token_idx_ptr = np.zeros(D + 1, dtype=np.int32)
    token_idx_ptr[1:] = np.cumsum(doc_term_matrix.sum(axis=1))
    word_per_topics = np.zeros(
        shape=(V, T), dtype=np.int32
    )  # отдельно каждое слово в каждом топике
    np.add.at(word_per_topics, (token_word_map, token_topic_map), 1)
    words_count_by_topic = np.bincount(
        token_topic_map, minlength=T
    )  # сколько слов в топике
    word_topic_by_document = np.zeros(
        shape=(D, T), dtype=np.int32
    )  # сколько всего слов в документе помечено темой
    np.add.at(word_topic_by_document, (token_document_map, token_topic_map), 1)
    for _iter in range(1, iteration):
        for token_idx in np.random.permutation(token_document_map.shape[0]):
            document = token_document_map[token_idx]
            word = token_word_map[token_idx]
            topic = token_topic_map[token_idx]

            # Forgetting...
            word_per_topics[word, topic] -= 1
            words_count_by_topic[topic] -= 1
            word_topic_by_document[document, topic] -= 1
            # new topic choosing
            new_topic_probability = (
                (word_per_topics[word].astype(np.float32) + beta)
                / (words_count_by_topic.astype(np.float32) + V * beta)
                * (word_topic_by_document[document].astype(np.float32) + alpha)
            )
            new_topic_probability /= new_topic_probability.sum()
            new_topic = np.random.choice(T, p=new_topic_probability)
            # membering new topic
            word_per_topics[word, new_topic] += 1
            words_count_by_topic[new_topic] += 1
            word_topic_by_document[document, new_topic] += 1

            token_topic_map[token_idx] = new_topic
    np.add.at(
        phi,
        (
            token_topic_map,
            token_word_map,
        ),
        1 / token_word_map.shape[0],
    )
    np.add.at(
        theta,
        (
            token_document_map,
            token_topic_map,
        ),
        1 / token_word_map.shape[0],
    )

    phi = phi.astype(np.float32)
    theta = theta.astype(np.float32)

    topic_sizes = words_count_by_topic.astype(np.float32)  # n_{z,·}
    phi = (word_per_topics.T + beta) / (topic_sizes[:, None] + V * beta)  # T×V

    doc_lens = word_topic_by_document.sum(axis=1, keepdims=True)
    theta = (word_topic_by_document + alpha) / (doc_lens + T * alpha)  # D×T
    return phi, theta
```

## Реализация

### Собственная реализация LDA
- Реализован алгоритм LDA с возможностью настройки параметров
- Поддержка различных форматов входных данных
- Интеграция с библиотекой Gensim для оценки качества
- Результаты когерентности: ~0.55

### Референсная реализация (scikit-learn)
- Использована библиотека `sklearn.decomposition.LatentDirichletAllocation`
- Оптимизированная реализация с использованием Cython
- Результаты когерентности: ~0.73

### Параметры модели
- doc_topic_prior (alpha) = 1.0
- topic_word_prior (beta) = 0.1
- random_state = 0

## Анализ различий в результатах

### Сравнение когерентности
- Собственная реализация: ~0.55
- scikit-learn: ~0.73
- Разница: ~0.18 (32.7% улучшение)

### Возможные причины различий
1. Оптимизации в scikit-learn:
   - Использование Cython для ускорения вычислений
   - Векторизованные операции
   - Оптимизированные структуры данных

2. Особенности реализации:
   - Разные стратегии инициализации
   - Различия в обработке краевых случаев
   - Особенности сходимости алгоритма

3. Технические аспекты:
   - Точность вычислений
   - Обработка разреженных матриц
   - Эффективность обновления статистик

## Оценка качества

### Метрики
- Использована метрика когерентности (coherence score)
- Оптимальное количество тем: 2
- Лучший score когерентности: ~0.7307 (scikit-learn)

### Сравнение с scikit-learn
- Реализована интеграция с `sklearn.decomposition.LatentDirichletAllocation`
- Значительное улучшение качества с референсной реализацией
- Поддержка тех же параметров и функциональности

## Ключевые особенности
1. Гибкая настройка гиперпараметров
2. Интеграция с популярными библиотеками (Gensim, scikit-learn)
3. Поддержка различных форматов данных
4. Возможность оценки качества модели
5. Сравнительный анализ с референсной реализацией

## Заключение
Реализованный алгоритм LDA показывает базовую работоспособность, но уступает оптимизированной реализации из scikit-learn по качеству когерентности (~0.55 против ~0.73). Основные различия связаны с оптимизациями в референсной реализации и особенностями обработки данных. Оптимальное количество тем для данного датасета - 2, что подтверждается высокой оценкой когерентности в реализации scikit-learn.
