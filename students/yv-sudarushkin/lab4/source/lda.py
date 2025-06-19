import numpy as np
import random
from tqdm import tqdm


class LDA:
    def __init__(self, n_topics=10, alpha=0.1, beta=0.1, n_iter=10):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter

    def preprocess(self, documents):
        """Превращает список слов в списки индексов + создает словарь"""
        word2id = {}
        id2word = {}
        corpus = []

        for doc in documents:
            word_ids = []
            for word in doc:
                if word not in word2id:
                    idx = len(word2id)
                    word2id[word] = idx
                    id2word[idx] = word
                word_ids.append(word2id[word])
            corpus.append(word_ids)

        self.vocab_size = len(word2id)
        self.word2id = word2id
        self.id2word = id2word
        return corpus

    def initialize(self, corpus):
        D = len(corpus)
        V = self.vocab_size
        K = self.n_topics

        # Матрицы подсчетов
        self.n_dk = np.zeros((D, K), dtype=int)       # [документ][тема]
        self.n_kv = np.zeros((K, V), dtype=int)       # [тема][слово]
        self.n_k = np.zeros(K, dtype=int)             # [тема]

        self.z_dn = []  # список тем для каждого слова в документе

        for d, doc in enumerate(corpus):
            current_doc_topics = []
            for word in doc:
                topic = random.randint(0, K - 1)
                current_doc_topics.append(topic)

                self.n_dk[d][topic] += 1
                self.n_kv[topic][word] += 1
                self.n_k[topic] += 1
            self.z_dn.append(current_doc_topics)

    def sample_topic(self, d, word_id):
        """Сэмплирует новую тему для данного слова в документе d"""
        probs = []
        for k in range(self.n_topics):
            p_w_t = (self.n_kv[k][word_id] + self.beta) / (self.n_k[k] + self.vocab_size * self.beta)
            p_t_d = self.n_dk[d][k] + self.alpha
            probs.append(p_w_t * p_t_d)
        probs = np.array(probs)
        probs /= probs.sum()
        return np.random.choice(self.n_topics, p=probs)

    def fit(self, raw_documents):
        corpus = self.preprocess(raw_documents)
        self.initialize(corpus)

        for it in tqdm(range(self.n_iter), desc="Training LDA"):
            for d, doc in enumerate(corpus):
                for i, word in enumerate(doc):
                    old_topic = self.z_dn[d][i]

                    # Удаляем
                    self.n_dk[d][old_topic] -= 1
                    self.n_kv[old_topic][word] -= 1
                    self.n_k[old_topic] -= 1

                    # Сэмплируем
                    new_topic = self.sample_topic(d, word)

                    # Обновляем
                    self.z_dn[d][i] = new_topic
                    self.n_dk[d][new_topic] += 1
                    self.n_kv[new_topic][word] += 1
                    self.n_k[new_topic] += 1

        return self

    def get_topic_words(self, top_n=10):
        result = []
        for k in range(self.n_topics):
            top_word_ids = self.n_kv[k].argsort()[::-1][:top_n]
            top_words = [self.id2word[i] for i in top_word_ids]
            result.append(top_words)
        return result
