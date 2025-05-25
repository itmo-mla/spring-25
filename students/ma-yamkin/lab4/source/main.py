from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import time
import numpy as np
from collections import defaultdict


def preprocess(text):
    """Функция предобработки текста:
    1. Удаление не-буквенных символов
    2. Приведение к нижнему регистру
    3. Удаление стоп-слов и коротких токенов
    4. Стемминг
    """
    # Токенизация и очистка
    tokens = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()

    # Фильтрация и стемминг
    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return processed_tokens


class GibbsSamplingLDA:
    def __init__(self, num_topics, alpha=0.1, beta=0.1, iterations=1000):
        """Инициализация параметров модели:
        - num_topics: количество тем
        - alpha: гиперпараметр распределения документ-темы (управление распределением тем в документах)
        - beta: гиперпараметр распределения тема-слова (управдение распределением тем в словах)
        - iterations: количество итераций Гиббс-сэмплинга
        """
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def fit(self, documents):
        """
        Смысл: мы инициализируем случайно слова-топики, документы-топики и темы, далее мы пробегаемся по каждому слову
        итерациями Гиббса и

        Обучение модели:
        1. Создание словаря
        2. Инициализация счетчиков
        3. Гиббс-сэмплинг
        4. Расчет итоговых распределений
        """
        # Создание словаря: слово -> индекс
        unique_words = set(word for doc in documents for word in doc)
        self.word_to_id = {word: idx for idx, word in enumerate(unique_words)}
        self.vocab_size = len(unique_words)

        # Инициализация счетчиков с добавлением сглаживания
        self.topic_word_counts = np.zeros((self.num_topics, self.vocab_size)) + self.beta  # слова-топики
        self.doc_topic_counts = np.zeros((len(documents), self.num_topics)) + self.alpha  # документы-топики
        self.total_topic_counts = np.zeros(self.num_topics) + self.vocab_size * self.beta  # количества тем

        # Инициализация случайных назначений тем для слов
        self.topic_assignments = []
        for doc_id, doc in enumerate(documents):
            doc_assignments = []
            for word in doc:
                word_id = self.word_to_id[word]
                # Случайное назначение темы при инициализации
                random_topic = np.random.randint(self.num_topics)
                doc_assignments.append(random_topic)

                # Обновление счетчиков
                self.topic_word_counts[random_topic, word_id] += 1
                self.doc_topic_counts[doc_id, random_topic] += 1
                self.total_topic_counts[random_topic] += 1
            self.topic_assignments.append(doc_assignments)

        # Процесс Гиббс-сэмплинга
        for iteration in range(self.iterations):
            for doc_id in range(len(documents)):
                for word_pos, word in enumerate(documents[doc_id]):
                    # здесь происходит удаление старой темы (поскольку считаем вероятность и нам необходимо исключить
                    # текущее слово, или его вероятность будет равна прежней) и добавление новой

                    word_id = self.word_to_id[word]
                    current_topic = self.topic_assignments[doc_id][word_pos]

                    # Удаление текущего назначения темы из счетчиков
                    self.topic_word_counts[current_topic, word_id] -= 1
                    self.doc_topic_counts[doc_id, current_topic] -= 1
                    self.total_topic_counts[current_topic] -= 1

                    # Вычисление вероятностей для новых тем (есть 10 тем для документы и 10 для текущего слова)
                    topic_probs = (
                            (self.doc_topic_counts[doc_id, :] + self.alpha) *
                            (self.topic_word_counts[:, word_id] + self.beta) /
                            (self.total_topic_counts + self.vocab_size * self.beta)
                    )

                    # Выбор новой темы из нормального распределения (используем распределение поскольку это
                    # позволяет избегать локальных оптимумов и лучуше исследовать пространство)
                    new_topic = np.random.normal(1, topic_probs / topic_probs.sum()).argmax()

                    # Обновление счетчиков с новым назначением темы
                    self.topic_word_counts[new_topic, word_id] += 1
                    self.doc_topic_counts[doc_id, new_topic] += 1
                    self.total_topic_counts[new_topic] += 1
                    self.topic_assignments[doc_id][word_pos] = new_topic

        # Расчет итоговых распределений
        self.word_distribution = self.topic_word_counts / self.total_topic_counts[:, np.newaxis]
        self.topic_distribution = self.doc_topic_counts / self.doc_topic_counts.sum(axis=1)[:, np.newaxis]


def calculate_topic_coherence(model, documents, top_words=10):
    """
    Вычисляет когерентность тем по метрике UMass

    Параметры:
    model - обученная модель LDA
    documents - список предобработанных документов
    top_words - количество топовых слов для оценки

    Возвращает:
    Среднюю когерентность по всем темам
    """
    # Создаем обратный словарь для преобразования ID в слова
    id_to_word = {v: k for k, v in model.word_to_id.items()}

    # Подсчет частот слов и совместных встречаемостей
    word_doc_count = defaultdict(int)  # D(w)
    co_occurrence_count = defaultdict(int)  # D(w_i, w_j)

    # Предварительный подсчет статистик
    for doc in documents:
        unique_words = set(doc)
        # Обновляем счетчики уникальных слов в документе
        for word in unique_words:
            word_doc_count[word] += 1

        # Обновляем счетчики совместных встречаемостей
        unique_words_list = list(unique_words)
        for i in range(len(unique_words_list)):
            for j in range(i + 1, len(unique_words_list)):
                pair = tuple(sorted([unique_words_list[i], unique_words_list[j]]))
                co_occurrence_count[pair] += 1

    coherence_scores = []

    for topic_id in range(model.num_topics):
        # Получаем топовые слова для темы
        topic_word_probs = model.word_distribution[topic_id]
        top_word_indices = np.argsort(topic_word_probs)[-top_words:]
        topic_words = [id_to_word[idx] for idx in top_word_indices]

        topic_score = 0
        valid_pairs = 0

        # Расчет попарной когерентности
        for i in range(1, len(topic_words)):
            for j in range(i):
                w1, w2 = topic_words[i], topic_words[j]
                pair = tuple(sorted([w1, w2]))

                # Частота совместной встречаемости
                D_pair = co_occurrence_count.get(pair, 0)

                # Частота более редкого слова (для сглаживания)
                D_w2 = word_doc_count.get(w2, 0)

                if D_w2 > 0:
                    # Формула когерентности UMass
                    topic_score += np.log((D_pair + 1) / D_w2)
                    valid_pairs += 1

        # Усреднение по валидным парам
        if valid_pairs > 0:
            coherence_scores.append(topic_score / valid_pairs)
        else:
            coherence_scores.append(0)

    return np.mean(coherence_scores)


# Основной исполняемый код
if __name__ == "__main__":
    newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    raw_documents = newsgroups.data[:500]  # Берем подмножество для демонстрации

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    processed_documents = [
        [stemmer.stem(word) for word in re.sub(r'[^a-zA-Z]', ' ', doc).lower().split()
         if word not in stop_words and len(word) > 2
         ] for doc in raw_documents]

    custom_lda = GibbsSamplingLDA(
        num_topics=10,
        alpha=0.1,
        beta=0.01,
        iterations=50
    )

    start_time = time.time()
    custom_lda.fit(processed_documents)
    custom_time = time.time() - start_time

    # Создаем единый векторйзер для сравнения
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    corpus = [' '.join(doc) for doc in processed_documents]
    X = vectorizer.fit_transform(corpus)

    sklearn_lda = LatentDirichletAllocation(
        n_components=10,
        learning_method='online',
        random_state=42,
        doc_topic_prior=0.1,  # alpha
        topic_word_prior=0.01,  # beta
        n_jobs=-1,
        max_iter=50
    )

    start_time = time.time()
    sklearn_lda.fit(X)
    sklearn_time = time.time() - start_time

    class SklearnLDAWrapper:
        def __init__(self, sklearn_model, feature_names):
            self.num_topics = sklearn_model.n_components
            self.word_distribution = sklearn_model.components_
            self.word_to_id = {word: idx for idx, word in enumerate(feature_names)}


    sklearn_wrapper = SklearnLDAWrapper(sklearn_lda, vectorizer.get_feature_names_out())

    custom_coherence = calculate_topic_coherence(custom_lda, processed_documents)
    sklearn_coherence = calculate_topic_coherence(sklearn_wrapper, processed_documents)

    print("\nРезультаты сравнения:")
    print(f"{'Метрика':<25} | {'Custom LDA':<12} | {'Scikit-learn':<12}")
    print("-" * 50)
    print(f"{'Время обучения (сек)':<25} | {custom_time:^12.2f} | {sklearn_time:^12.2f}")
    print(f"{'Когерентность тем':<25} | {custom_coherence:^12.4f} | {sklearn_coherence:^12.4f}")
