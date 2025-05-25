import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class LDA:

    def __init__(self, num_topics, alpha=0.1, beta=0.01, max_iter=100):
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter


    def fit(self, raw_documents):
        self._preprocess(raw_documents)
        self._initialize()

        for _ in range(self.max_iter):
            self._e_step()
            self._m_step()

        self._estimate_parameters()


    def _preprocess(self, raw_documents):
        vectorizer = CountVectorizer(lowercase=True, stop_words='english')
        X = vectorizer.fit_transform(raw_documents)

        self.vocab = vectorizer.vocabulary_
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}

        self.documents = []
        for doc_id in range(X.shape[0]):
            word_ids = X[doc_id].nonzero()[1]
            counts = X[doc_id].data
            self.documents.append(list(zip(word_ids, counts)))


    def _initialize(self):
        self.V = len(self.vocab)
        self.D = len(self.documents)

        self.phi = np.random.dirichlet([self.beta] * self.V, size=self.num_topics)
        self.gamma = np.random.dirichlet([self.alpha] * self.num_topics, size=self.D)


    def _e_step(self):
        self.z = [np.zeros((len(doc), self.num_topics)) for doc in self.documents]
        for d, doc in enumerate(self.documents):
            gamma_d = np.zeros(self.num_topics)
            for n, (w, count) in enumerate(doc):
                topic_probs = self.gamma[d] * self.phi[:, w]
                topic_probs /= topic_probs.sum()
                self.z[d][n] = topic_probs
                gamma_d += count * topic_probs
            self.gamma[d] = self.alpha + gamma_d


    def _m_step(self):
        self.phi = np.zeros((self.num_topics, self.V))
        for d, doc in enumerate(self.documents):
            for n, (w, count) in enumerate(doc):
                self.phi[:, w] += count * self.z[d][n]
        self.phi += self.beta
        self.phi /= self.phi.sum(axis=1, keepdims=True)


    def _estimate_parameters(self):
        self.theta = self.gamma / np.sum(self.gamma, axis=1, keepdims=True)


    def get_top_words(self, n_words=5):
        top_words = []
        for k in range(self.num_topics):
            top_ids = np.argsort(self.phi[k])[-n_words:][::-1]
            top_words.append([self.inv_vocab[i] for i in top_ids])
        return top_words