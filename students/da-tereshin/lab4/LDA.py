import numpy as np
from scipy.special import digamma


class LDA:
    """Реализация алгоритма LDA с вариационным байесовским выводом"""

    def __init__(self, n_components: int, max_iter: int = 10, alpha: float = 0.1,
                 beta: float = 0.1, tol: float = 1e-3, random_state: int = 42) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta  # в литературе еще eta
        self.tol = tol
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)

    @property
    def components_(self):
        """Возвращает нормированную матрицу 'тема × слово' для вычисления когерентности"""
        if self.lambda_ is None:
            raise ValueError("Модель ещё не обучена!")

        return self.lambda_ / np.sum(self.lambda_, axis=1, keepdims=True)

    def _init_params(self, n_docs, n_words):
        """Инициализация вариационных параметров"""

        # gamma[d][k] ~ Dir(alpha) – распределение тем в документе (в русской литературе theta_d)
        self.gamma = self.rng.gamma(100., 1. / 100., size=(n_docs, self.n_components)) + self.alpha

        # lambda_[k][w] ~ Dir(beta) – распределение слов в теме (в русской литературе phi_t)
        self.lambda_ = self.rng.gamma(100., 1. / 100., size=(self.n_components, n_words)) + self.beta

        # Вероятность темы k для слова n в документе d, генерируется из dirichlet
        # phi[d][n][k] — заменяется на матричные операции, поэтому отдельно не храним

    def _e_step(self, X):
        """E-шаг: обновление gamma (распределение тем в документах)."""

        # E[log(theta_dk)] = digamma(gamma_dk) - digamma(sum_l gamma_dl)
        E_log_theta = digamma(self.gamma) - digamma(np.sum(self.gamma, axis=1, keepdims=True))

        # E[log(beta_kw)] = digamma(lambda_kw) - digamma(sum_v lambda_kv)
        E_log_beta = digamma(self.lambda_) - digamma(np.sum(self.lambda_, axis=1, keepdims=True))

        # phi[d][w][k] ∝ exp(E_log_theta[d][k] + E_log_beta[k][w])
        phi = np.exp(E_log_theta[:, np.newaxis, :] + E_log_beta.T[np.newaxis, :, :])
        phi /= np.sum(phi, axis=2, keepdims=True)

        self.gamma = self.alpha + np.sum(X[:, :, np.newaxis] * phi, axis=1)

    def _m_step(self, X):
        """M-шаг: обновление lambda (распределение слов в темах)."""

        E_log_theta = digamma(self.gamma) - digamma(np.sum(self.gamma, axis=1, keepdims=True))
        E_log_beta = digamma(self.lambda_) - digamma(np.sum(self.lambda_, axis=1, keepdims=True))

        # phi[d][w][k] ∝ exp(E_log_theta[d][k] + E_log_beta[k][w])
        phi = np.exp(E_log_theta[:, np.newaxis, :] + E_log_beta.T[np.newaxis, :, :])
        phi /= np.sum(phi, axis=2, keepdims=True)

        # lambda[k][w] = beta + sum_d X[d][w] * phi[d][w][k]
        self.lambda_ = self.beta + np.sum(X[:, :, np.newaxis] * phi, axis=0).T  # n_topics × n_words

    def fit(self, X: np.ndarray) -> None:
        # Если разреженная, преобразуем в плотную
        X = X.toarray() if hasattr(X, 'toarray') else np.array(X)

        # Количество документов и размер словаря
        self.n_docs, self.n_words = X.shape

        # Инициализация
        self._init_params(self.n_docs, self.n_words)

        # Variational Bayes EM-алгоритм
        prev_lambda = np.zeros_like(self.lambda_)
        for _ in range(self.max_iter):
            # E-шаг: обновления gamma
            self._e_step(X)
            # M-шаг: обновление lambda
            self._m_step(X)

            # Проверка на сходимость
            if np.linalg.norm(self.lambda_ - prev_lambda) < self.tol:
                print(f"Сходимость достигнута на итерации {i}")
                break
            prev_lambda = self.lambda_.copy()

    def transform(self, X_new):
        X_new = X_new.toarray() if hasattr(X_new, 'toarray') else np.array(X_new)
        n_new_docs, n_words = X_new.shape

        gamma_new = np.random.gamma(100., 1. / 100., size=(n_new_docs, self.n_components)) + self.alpha

        # E-шаг (только для новых документов, lambda_ фиксировано)
        for _ in range(self.max_iter):
            E_log_theta = digamma(gamma_new) - digamma(np.sum(gamma_new, axis=1, keepdims=True))
            E_log_beta = digamma(self.lambda_) - digamma(np.sum(self.lambda_, axis=1, keepdims=True))

            # Обновление gamma_new
            phi = np.exp(E_log_theta[:, np.newaxis, :] + E_log_beta.T[np.newaxis, :, :])
            phi /= np.sum(phi, axis=2, keepdims=True)
            gamma_new = self.alpha + np.sum(X_new[:, :, np.newaxis] * phi, axis=1)

        return gamma_new / np.sum(gamma_new, axis=1, keepdims=True)

    def get_topics(self, feature_names, top_n=10):
        """Возвращает top_n слов для каждой темы."""
        topics = []
        for k in range(self.n_components):
            top_words_idx = np.argsort(-self.lambda_[k])[:top_n]
            topic_words = [feature_names[idx] for idx in top_words_idx]
            topics.append(topic_words)
        return topics
