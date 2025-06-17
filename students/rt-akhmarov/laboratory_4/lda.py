import re

import gensim
import numpy as np

from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer

from scipy.sparse import csr_matrix
from scipy.special import digamma

class LdaPreprocessor:
    def __init__(self,
                 max_features: int = 1000,
                 stop_words='english',
                 max_df: float = 0.95,
                 min_df: int = 2,
                 ngram_range: tuple = (1, 1)):

        self.stop_words = stop_words 

        self.vectorizer = CountVectorizer(
            preprocessor=self._clean_text,
            stop_words=self.stop_words,
            max_features=max_features,
            max_df=max_df,
            min_df=min_df,
            ngram_range=ngram_range,
            token_pattern=r'(?u)\b\w\w+\b'
        )

    def _clean_text(self, doc: str) -> str:
        doc = doc.lower()
        return re.sub(r'[^a-z\s]', ' ', doc)

    def fit(self, X):
        return self.vectorizer.fit_transform(X)

    def transform(self, X):
        return self.vectorizer.transform(X)

    def fit_transform(self, X):
        return self.fit(X)

    @property
    def feature_names_(self):
        return self.vectorizer.get_feature_names_out()


class LdaCoherenceEvaluator:
    def __init__(self,
                 docs: list[str],
                 processor: LdaPreprocessor):

        self.analyzer = processor.vectorizer.build_analyzer()
        self.feature_names = processor.feature_names_
        self.tokenized_texts = [self.analyzer(doc) for doc in docs]
        self.dictionary = gensim.corpora.Dictionary(self.tokenized_texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.tokenized_texts]

    def coherence(self, model, mode:str = 'c_v', per_topic:bool = False) -> float:
        topics = []

        for _, comp in enumerate(model.components_):
            top_indices = comp.argsort()[-10:]
            topics.append([self.feature_names[i] for i in top_indices])

        coh_model = CoherenceModel(
            topics=topics,
            dictionary=self.dictionary,
            texts=self.tokenized_texts if mode == 'c_v' else None,
            corpus=self.corpus if mode == 'u_mass' else None,
            coherence=mode
        )
        if per_topic:
            return coh_model.get_coherence_per_topic()
        else:
            return coh_model.get_coherence()
    

class LatentDirichletAllocation:
    def __init__(self,
                 n_topics: int = 10,
                 max_iter: int = 10,
                 alpha: float = None,
                 eta: float = None,
                 tol: float = 1e-3,
                 random_state: int = None,
                 verbose: int = 0):
        """
        n_topics      — число тем (K).
        max_iter      — максимальное число итераций вариационного вывода.
        alpha         — параметр Дирихле для распределения θ (документ–тема).
        eta           — параметр Дирихле для распределения β (тема–слово).
        tol           — порог сходимости (напр. по ELBO).
        random_state  — seed для генератора случайных чисел.
        verbose       — уровень логирования.
        """

        self.n_topics = n_topics
        self.max_iter = max_iter
        self.alpha    = alpha    
        self.eta      = eta     
        self.tol      = tol
        self.verbose  = verbose

        if isinstance(random_state, (int, np.integer)):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state or np.random.RandomState()

        self.components_ = None
        self.exp_dirichlet_component_ = None
        self.doc_topic_distr_ = None

        self.is_fitted_ = False

        
    def fit(self, X_counts):
        """
        Обучаем LDA по count-матрице X_counts (shape [D, V]).
        """
        X = csr_matrix(X_counts, dtype=int)
        D, V = X.shape
        K = self.n_topics

        if self.alpha is None:
            self.alpha = 1.0 / K
        if self.eta is None:
            self.eta = 1.0 / K

        alpha_vec = np.full(K, self.alpha)
        eta = self.eta

        self.components_ = (
            self.random_state.gamma(100.0, 1.0 / 100.0, size=(K, V))
            + eta
        )

        self.exp_dirichlet_component_ = self._compute_expElogbeta(self.components_)

        self.doc_topic_distr_ = np.zeros((D, K))

        for iteration in range(self.max_iter):
            if self.verbose:
                print(f"[LDA] Iteration {iteration+1}/{self.max_iter}")

            sstats = np.zeros_like(self.components_)

            for d in range(D):
                row = X[d]
                if row.nnz == 0:
                    self.doc_topic_distr_[d] = alpha_vec / alpha_vec.sum()
                    continue

                indices = row.indices   
                counts  = row.data      
                N_d = counts.sum()

                gamma_d = alpha_vec + (N_d / K)

                for _ in range(100):  
                    last_gamma = gamma_d.copy()

                    E_log_theta = digamma(gamma_d) - digamma(gamma_d.sum())
                    exp_E_log_theta = np.exp(E_log_theta)

                    phi_v = (
                        exp_E_log_theta[:, None]
                        * self.exp_dirichlet_component_[:, indices]
                    )  # shape (K, len(indices))
                    phi_v /= phi_v.sum(axis=0)[None, :]

                    gamma_d = alpha_vec + (phi_v * counts).sum(axis=1)

                    if np.mean(np.abs(gamma_d - last_gamma)) < self.tol:
                        break

                self.doc_topic_distr_[d] = gamma_d / gamma_d.sum()

                sstats[:, indices] += phi_v * counts

            self.components_ = sstats + eta
            self.exp_dirichlet_component_ = self._compute_expElogbeta(self.components_)

        self.is_fitted_ = True
        return self
    
    def _compute_expElogbeta(self, comp):
        logt = digamma(comp) - digamma(np.sum(comp, axis=1))[:, None]
        return np.exp(logt)
    
