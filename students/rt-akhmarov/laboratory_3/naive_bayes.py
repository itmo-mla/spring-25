import numpy as np
from sklearn.base import BaseEstimator

class NaiveBayes(BaseEstimator):
    def __init__(self, alpha:float = 1.0):
        self.class_log_proba_ = {}
        self.feature_log_proba_ = {}
        self.feature_log_proba_matrix = None
        self.class_log_proba_matrix = None
        self.vocab_size = 0
        self.classes_ = None
        self.alpha = alpha

    def fit(self, X, y):
        n_docs, n_features = X.shape
        self.vocab_size = n_features
        self.classes_ = np.unique(y)
        words_count_ = self._calculate_word_counts_by_class(X, y)
        total_words_ = {cls: words_count_[cls].sum() for cls in self.classes_}

        for cls in self.classes_:
            self.feature_log_proba_[cls] = np.log(
                (words_count_[cls] + self.alpha) / (total_words_[cls] + self.alpha * self.vocab_size)
            )
            self.class_log_proba_[cls] = np.log(
                np.sum(y == cls) / n_docs
            )

        self.feature_log_proba_matrix = np.vstack([self.feature_log_proba_[cls] for cls in self.classes_])
        self.class_log_proba_matrix = np.array([self.class_log_proba_[cls] for cls in self.classes_])

        return self

    def predict(self, X):
        log_likelihood = X.dot(self.feature_log_proba_matrix.T)
        log_posterior = log_likelihood + self.class_log_proba_matrix

        cls_idx = np.argmax(log_posterior, axis=-1)

        return np.array([self.classes_[idx] for idx in cls_idx])

    def _calculate_word_counts_by_class(self, X, y):
        labels = np.unique(y) if self.classes_ is None else self.classes_
        words_count = {}

        for cls in labels:
            idx = np.where(y == cls)[0]
            X_cls = X[idx]
            words_count[cls] = np.asarray(X_cls.sum(axis=0)).flatten()

        return words_count