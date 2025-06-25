import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from typing import Sequence
from sklearn.datasets import fetch_20newsgroups


class GibbsLDA:

    def __init__(
        self,
        n_topics: int,
        alpha: float | Sequence[float] = 0.1,
        beta: float = 0.01,
        n_iter: int = 50,
        random_state: int | None = None,
    ) -> None:
        self.n_topics = n_topics
        self.alpha = (
            np.full(n_topics, alpha, dtype=np.float64)
            if np.isscalar(alpha)
            else np.asarray(alpha, dtype=np.float64)
        )
        self.beta = beta
        self.n_iter = n_iter
        self.random_state = random_state

        self.topic_word_ = None  #  (n_topics, vocab)
        self.doc_topic_ = None  #  (n_docs, n_topics)
        self._assignments: list[list[int]] = []

    def fit(self, X: sparse.spmatrix) -> "GibbsLDA":
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = sparse.csr_matrix(
            X, dtype=np.int32
        )  # гарантируем формат, потому что без это ОНО ПАДАЕТ
        n_docs, vocab_size = X.shape

        self.topic_word_ = np.zeros((self.n_topics, vocab_size), dtype=np.int32)
        self.doc_topic_ = np.zeros((n_docs, self.n_topics), dtype=np.int32)
        self._assignments = []

        # Рандооом
        for d in range(n_docs):
            row = X.indices[X.indptr[d] : X.indptr[d + 1]]
            counts = X.data[X.indptr[d] : X.indptr[d + 1]]

            doc_topics: list[int] = []
            for w, c in zip(row, counts):
                for _ in range(c):
                    t = np.random.randint(self.n_topics)
                    doc_topics.append(t)
                    self.topic_word_[t, w] += 1
                    self.doc_topic_[d, t] += 1
            self._assignments.append(doc_topics)

        # Гиббс
        beta_sum = vocab_size * self.beta
        for _ in range(self.n_iter):
            for d in range(n_docs):
                row = X.indices[X.indptr[d] : X.indptr[d + 1]]
                counts = X.data[X.indptr[d] : X.indptr[d + 1]]

                token_idx = 0
                for w, c in zip(row, counts):
                    for _ in range(c):
                        old_t = self._assignments[d][token_idx]

                        self.topic_word_[old_t, w] -= 1
                        self.doc_topic_[d, old_t] -= 1

                        # p(t)
                        left = self.doc_topic_[d] + self.alpha
                        right = (self.topic_word_[:, w] + self.beta) / (
                            self.topic_word_.sum(axis=1) + beta_sum
                        )
                        probs = left * right
                        probs /= probs.sum()

                        new_t = np.random.choice(self.n_topics, p=probs)

                        self._assignments[d][token_idx] = new_t
                        self.topic_word_[new_t, w] += 1
                        self.doc_topic_[d, new_t] += 1

                        token_idx += 1

        self.topic_word_ = (self.topic_word_ + self.beta) / (
            self.topic_word_.sum(axis=1, keepdims=True) + beta_sum
        )
        self.doc_topic_ = (self.doc_topic_ + self.alpha) / (
            self.doc_topic_.sum(axis=1, keepdims=True) + self.alpha.sum()
        )
        return self

    def top_words(self, vectorizer: CountVectorizer, n: int = 10) -> list[list[str]]:
        vocab = np.array(vectorizer.get_feature_names_out())
        idx = np.argsort(self.topic_word_, axis=1)[:, : -n - 1 : -1]
        return vocab[idx].tolist()

    def transform(self, X: sparse.spmatrix) -> np.ndarray:
        return self.doc_topic_


import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from scipy import sparse


def calc_perplexity(phi, theta, X):
    X = sparse.csr_matrix(X)
    loglik = 0.0
    words = 0
    for d in range(X.shape[0]):
        row_idx = X.indices[X.indptr[d] : X.indptr[d + 1]]
        cnts = X.data[X.indptr[d] : X.indptr[d + 1]]
        probs = theta[d] @ phi[:, row_idx]
        probs = np.maximum(probs, 1e-12)
        loglik += np.sum(cnts * np.log(probs))
        words += np.sum(cnts)
    return np.exp(-loglik / words)


newsgroups = fetch_20newsgroups(
    subset="test",
    remove=("headers", "footers", "quotes"),
)
texts = newsgroups.data[:1000]

print(f"Документов: {len(texts)}")


train_texts, test_texts = train_test_split(texts, test_size=0.3, random_state=42)

vectorizer = CountVectorizer(stop_words="english", max_features=1000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)


n_topics = 5
n_iter = 30


start = time.perf_counter()
gibbs = GibbsLDA(n_topics=n_topics, n_iter=n_iter, random_state=0)
gibbs.fit(X_train)
time_gibbs = time.perf_counter() - start
perp_gibbs = calc_perplexity(gibbs.topic_word_, gibbs.doc_topic_, X_test)


start = time.perf_counter()
sk_lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=n_iter,
    learning_method="batch",
    random_state=0,
)
sk_lda.fit(X_train)
time_sk = time.perf_counter() - start
perp_sk = np.exp(-sk_lda.score(X_test) / X_test.sum())


results = pd.DataFrame(
    {
        "Model": ["GibbsLDA", "sklearn LDA"],
        "Training time (s)": [time_gibbs, time_sk],
        "Perplexity": [perp_gibbs, perp_sk],
    }
)
print(results)


for k, words in enumerate(gibbs.top_words(vectorizer, n=10), 1):
    print(f"\nTopic {k}: " + ", ".join(words))

theta = gibbs.transform(X_test)
for d in range(2):
    top = np.argsort(theta[d])[-3:][::-1]
    print(
        f"\nДокумент {d}:  самые сильные темы → "
        + ", ".join(f"{t} ({theta[d,t]:.2f})" for t in top)
    )
