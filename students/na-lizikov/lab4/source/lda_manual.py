import numpy as np
import random

class LDAManual:
    def __init__(self, n_topics=10, n_iter=100, alpha=0.1, beta=0.01, random_state=42):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.topic_word_ = None
        self.doc_topic_ = None
        self.topic_word_dist_ = None
        self.doc_topic_dist_ = None
        self.top_words_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        D, W = X.shape
        K = self.n_topics
        n_iter = self.n_iter
        alpha = self.alpha
        beta = self.beta

        # X в список слов для каждого документа
        docs = []
        for d in range(D):
            doc = []
            row = X[d].toarray().flatten()
            for w, count in enumerate(row):
                doc += [w] * count
            docs.append(doc)

        # Инициализация
        z_dn = []
        doc_topic = np.zeros((D, K), dtype=int)
        topic_word = np.zeros((K, W), dtype=int)
        topic_count = np.zeros(K, dtype=int)

        for d, doc in enumerate(docs):
            z_n = []
            for w in doc:
                z = np.random.randint(K)
                z_n.append(z)
                doc_topic[d, z] += 1
                topic_word[z, w] += 1
                topic_count[z] += 1
            z_dn.append(z_n)

        # Гиббсово сэмплирование
        for it in range(n_iter):
            for d, doc in enumerate(docs):
                for n, w in enumerate(doc):
                    z = z_dn[d][n]
                    doc_topic[d, z] -= 1
                    topic_word[z, w] -= 1
                    topic_count[z] -= 1

                    # Вычисление вероятностей для каждой темы
                    p_z = (doc_topic[d] + alpha) * (topic_word[:, w] + beta) / (topic_count + W * beta)
                    p_z = p_z / p_z.sum()
                    z_new = np.random.choice(K, p=p_z)

                    z_dn[d][n] = z_new
                    doc_topic[d, z_new] += 1
                    topic_word[z_new, w] += 1
                    topic_count[z_new] += 1
            if (it+1) % 10 == 0:
                print(f"Итерация {it+1}/{n_iter} завершена")

        self.topic_word_ = topic_word
        self.doc_topic_ = doc_topic
        self.topic_word_dist_ = (topic_word + beta) / (topic_word.sum(axis=1)[:, None] + W * beta)
        self.doc_topic_dist_ = (doc_topic + alpha) / (doc_topic.sum(axis=1)[:, None] + K * alpha)

    def get_top_words(self, feature_names, n_top_words=10):
        top_words = []
        for topic_idx, topic in enumerate(self.topic_word_dist_):
            top = topic.argsort()[::-1][:n_top_words]
            top_words.append([feature_names[i] for i in top])
        self.top_words_ = top_words
        return top_words 