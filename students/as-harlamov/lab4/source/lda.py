import numpy as np


class LDA:
    def __init__(self, n_topics, alpha=0.1, beta=0.01, n_iter=20):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter

        self.vocab_size = None
        self.doc_topic_counts = None
        self.topic_word_counts = None
        self.topic_counts = None
        self.assignments = None

    def fit(self, corpus, vocab_size):
        self.vocab_size = vocab_size
        self.doc_topic_counts = np.zeros((len(corpus), self.n_topics))
        self.topic_word_counts = np.zeros((self.n_topics, vocab_size))
        self.topic_counts = np.zeros(self.n_topics)
        self.assignments = []

        for d_idx, doc in enumerate(corpus):
            topics = []
            for word in doc:
                p_z = (
                    (self.topic_word_counts[:, word] + self.beta) *
                    (self.doc_topic_counts[d_idx] + self.alpha) /
                    (self.topic_counts + self.beta * vocab_size)
                )
                topic = np.random.choice(self.n_topics, p=p_z / p_z.sum())
                topics.append(topic)
                self.doc_topic_counts[d_idx, topic] += 1
                self.topic_word_counts[topic, word] += 1
                self.topic_counts[topic] += 1
            self.assignments.append(np.array(topics))

        for _ in range(self.n_iter):
            for d_idx, doc in enumerate(corpus):
                for i, word in enumerate(doc):
                    topic = self.assignments[d_idx][i]
                    self.doc_topic_counts[d_idx, topic] -= 1
                    self.topic_word_counts[topic, word] -= 1
                    self.topic_counts[topic] -= 1

                    p_z = (
                        (self.topic_word_counts[:, word] + self.beta) *
                        (self.doc_topic_counts[d_idx] + self.alpha) /
                        (self.topic_counts + self.beta * vocab_size)
                    )
                    new_topic = np.random.choice(self.n_topics, p=p_z / p_z.sum())

                    self.doc_topic_counts[d_idx, new_topic] += 1
                    self.topic_word_counts[new_topic, word] += 1
                    self.topic_counts[new_topic] += 1
                    self.assignments[d_idx][i] = new_topic

    def get_topics(self, top_n=10):
        topic_words = []
        for k in range(self.n_topics):
            top_words = np.argsort(self.topic_word_counts[k])[-top_n:][::-1]
            topic_words.append(top_words)
        return topic_words

    def get_vocabulary(self, index_to_word):
        return [[index_to_word[i] for i in topic] for topic in self.get_topics()]
