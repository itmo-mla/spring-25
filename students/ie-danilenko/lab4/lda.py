import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class LDA:
    def __init__(self, n_components, alpha=0.1, beta=0.01, max_iter=50, random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter 
        self.random_state = random_state
        self.components_ = None
        self.topic_word_counts_ = None
        self.doc_topic_counts_ = None

    def fit(self, X, y=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        self.vocab_size = n_features

        # Инициализируем параметры
        self.topic_word_counts_ = np.zeros((self.n_components, n_features))
        self.doc_topic_counts_ = np.zeros((n_samples, self.n_components))

        # Хранение назначенных тем
        self.assignments_ = []

        # Проходим по всем документам
        for i in range(n_samples):
            row = X[i].toarray()[0] if hasattr(X, "toarray") else X[i]
            assignments = []
            for j in range(n_features):
                count = int(row[j])
                for _ in range(count):
                    topic = np.random.randint(0, self.n_components)
                    self.doc_topic_counts_[i, topic] += 1
                    self.topic_word_counts_[topic, j] += 1
                    assignments.append(topic)
            self.assignments_.append(assignments)

        # Гиббсовский отбор
        for _ in range(self.max_iter):
            for i in range(n_samples):
                row = X[i].toarray()[0] if hasattr(X, "toarray") else X[i]
                word_ids = []
                counts = []
                for j in range(n_features):
                    cnt = int(row[j])
                    if cnt > 0:
                        word_ids.extend([j] * cnt)
                        counts.append((j, cnt))

                for idx, (word_id, _) in enumerate(counts):
                    old_topic = self.assignments_[i][idx]

                    # Уменьшаем счетчики
                    self.doc_topic_counts_[i, old_topic] -= 1
                    self.topic_word_counts_[old_topic, word_id] -= 1

                    # Вычисляем условную вероятность
                    p_topic = (
                        (self.doc_topic_counts_[i] + self.alpha) *
                        (self.topic_word_counts_[:, word_id] + self.beta) /
                        (self.topic_word_counts_.sum(axis=1) + self.vocab_size * self.beta)
                    )

                    # Защита от нулей и NaN
                    p_topic = np.clip(p_topic, a_min=1e-12, a_max=None)
                    p_topic /= p_topic.sum()

                    new_topic = np.random.choice(self.n_components, p=p_topic)

                    # Обновляем
                    self.doc_topic_counts_[i, new_topic] += 1
                    self.topic_word_counts_[new_topic, word_id] += 1
                    self.assignments_[i][idx] = new_topic

        # Нормализуем компоненты
        self.components_ = self.topic_word_counts_.copy()
        for k in range(self.n_components):
            self.components_[k] /= self.components_[k].sum()

        return self

    def get_topics(self, vectorizer, num_words=10):
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for k in range(self.n_components):
            top_words_idx = self.components_[k].argsort()[-num_words:][::-1]
            topic_words = [feature_names[i] for i in top_words_idx]
            topics.append(topic_words)
        return topics


if __name__ == "__main__":
    from read import read_texts

    documents = read_texts('data/cleaned_texts.csv')
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(documents)

    lda = LDA(n_components=3, max_iter=50, random_state=42)
    lda.fit(X)

    topics = lda.get_topics(vectorizer, 10)
    for i, t in enumerate(topics):
        print(f"Topic {i+1}:", ", ".join(t))