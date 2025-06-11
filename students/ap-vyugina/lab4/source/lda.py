import numpy as np
from scipy import sparse
from typing import List
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary


def normalize(array, axis):
    return array / np.sum(array, axis=axis, keepdims=True)


class LDA:
    def __init__(self, n_topics: int, alpha: float = 0.1, beta: float = 0.1, max_iter: int = 100):
        """
        Initialize LDA model.
        
        Args:
            n_topics: Number of topics to extract
            alpha: Dirichlet prior for document-topic distribution
            beta: Dirichlet prior for topic-word distribution
            max_iter: Maximum number of iterations for EM algorithm
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        
        # Model parameters
        self.phi_wt = None # p(w|t) - count of words assigned to topics
        self.theta_td = None # p(t|d) - count of topics in documents
        self.topic_assignments = None # p_tdw = p(t|d,w) - Topic assignments for each word
        self.n_words = None # Number of features/words/tokens
        self.n_docs = None # Number of documents
        
        
    def _initialize(self, doc_term_matrix: sparse.csr_matrix):
        """Initialize model parameters."""
        self.n_docs, self.n_words = doc_term_matrix.shape
        
        n_wt = np.zeros((self.n_words, self.n_topics))
        n_td = np.zeros((self.n_docs, self.n_topics))
        
        self.topic_assignments = np.zeros((self.n_docs, self.n_words, self.n_topics))
        
        # For each document and word
        for d in range(self.n_docs):
            for w in range(self.n_words):
                if doc_term_matrix[d, w] > 0:
                    # Randomly assign topic
                    topic = np.random.randint(0, self.n_topics)
                    self.topic_assignments[d, w, topic] = 1
                    # Update counts
                    n_wt[w, topic] += doc_term_matrix[d, w]
                    n_td[d, topic] += doc_term_matrix[d, w]
        
        self.phi_wt = normalize(n_wt + self.beta - 1, axis=0)
        self.theta_td = normalize(n_td + self.alpha - 1, axis=1)

    
    def _e_step(self, doc_term_matrix: sparse.csr_matrix):
        """Perform E-step of EM algorithm."""
        for d in range(self.n_docs):
            for w in range(self.n_words):
                if doc_term_matrix[d, w] > 0: # if the word is in current document
                    # Calculate topic probabilities using already normalized distributions
                    topic_probs = self.phi_wt[w, :] * self.theta_td[d, :]
                    # Normalize
                    topic_probs /= np.sum(topic_probs)
                    self.topic_assignments[d, w, :] = topic_probs
        self.topic_assignments = normalize(self.topic_assignments, axis=2)

    def _m_step(self, doc_term_matrix: sparse.csr_matrix):
        """Perform M-step of EM algorithm."""
        # Initialize counts
        n_wt = np.zeros((self.n_words, self.n_topics))
        n_td = np.zeros((self.n_docs, self.n_topics))
        
        # Update counts based on expected assignments
        for d in range(self.n_docs):
            for w in range(self.n_words):
                if doc_term_matrix[d, w] > 0: # if the word is in current document
                    # Update counts using expected assignments
                    n_wt[w, :] += doc_term_matrix[d, w] * self.topic_assignments[d, w, :]
                    n_td[d, :] += doc_term_matrix[d, w] * self.topic_assignments[d, w, :]
        
        # Normalize the counts to get proper probability distributions
        self.phi_wt = normalize(n_wt + self.beta - 1, axis=0)
        self.theta_td = normalize(n_td + self.alpha - 1, axis=1)

    
    def fit(self, doc_term_matrix: sparse.csr_matrix):
        """
        doc_term_matrix: Sparse matrix of shape (n_docs, n_features) from CountVectorizer
        """
        self._initialize(doc_term_matrix)
        prev_phi_wt = self.phi_wt.copy()
        
        for i in range(self.max_iter):

            self._e_step(doc_term_matrix)
            self._m_step(doc_term_matrix)
            # print(self.phi_wt)
            error = np.sum(np.abs(self.phi_wt - prev_phi_wt))
            print(f"Iter {i+1}: {error:.4f}")
            if error < 0.05: break
            prev_phi_wt = self.phi_wt
            # if i > 0 and i % 10 == 0: print(i)
    
    def get_topic_word_distribution(self) -> np.ndarray:
        return self.phi_wt
    
    def get_document_topic_distribution(self) -> np.ndarray:
        return self.theta_td
    