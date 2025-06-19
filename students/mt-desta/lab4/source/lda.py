import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Union
import random
from scipy.sparse import csr_matrix

class LDA:
    def __init__(self, n_topics: int, alpha: float = 0.1, beta: float = 0.1, max_iter: int = 100):
        """
        Initialize Latent Dirichlet Allocation model.
        
        Args:
            n_topics (int): Number of topics to discover
            alpha (float): Dirichlet prior for document-topic distribution
            beta (float): Dirichlet prior for topic-word distribution
            max_iter (int): Maximum number of iterations for Gibbs sampling
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        
        # Model parameters
        self.word_topic_counts = None  # Count of words assigned to topics
        self.doc_topic_counts = None   # Count of topics in documents
        self.topic_counts = None       # Total count of words in each topic
        self.doc_lengths = None        # Length of each document
        
        # Vocabulary and document mappings
        self.vocab = None              # Vocabulary mapping
        self.inv_vocab = None          # Inverse vocabulary mapping
        self.documents = None          # Processed documents
        
    def _initialize_counts(self, X: Union[List[List[str]], csr_matrix], feature_names: List[str] = None):
        """Initialize count matrices for Gibbs sampling."""
        if isinstance(X, csr_matrix):
            # Handle sparse matrix input
            n_docs, n_words = X.shape
            self.vocab = {word: idx for idx, word in enumerate(feature_names)}
            self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
            
            # Initialize count matrices
            self.word_topic_counts = np.zeros((n_words, self.n_topics))
            self.doc_topic_counts = np.zeros((n_docs, self.n_topics))
            self.topic_counts = np.zeros(self.n_topics)
            self.doc_lengths = np.array(X.sum(axis=1)).flatten()
            
            # Convert sparse matrix to list of word lists
            self.documents = []
            self.word_topics = []
            
            for doc_idx in range(n_docs):
                # Get non-zero elements for this document
                doc_words = X[doc_idx].nonzero()[1]
                doc_counts = X[doc_idx].data
                
                # Create word list with repetitions
                doc = []
                for word_idx, count in zip(doc_words, doc_counts):
                    doc.extend([word_idx] * int(count))
                
                self.documents.append(doc)
                doc_topics = []
                
                # Randomly assign topics
                for word_idx in doc:
                    topic = random.randint(0, self.n_topics - 1)
                    doc_topics.append((word_idx, topic))
                    
                    # Update counts
                    self.word_topic_counts[word_idx, topic] += 1
                    self.doc_topic_counts[doc_idx, topic] += 1
                    self.topic_counts[topic] += 1
                
                self.word_topics.append(doc_topics)
        else:
            # Handle list of word lists input
            unique_words = set(word for doc in X for word in doc)
            self.vocab = {word: idx for idx, word in enumerate(unique_words)}
            self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
            
            # Initialize count matrices
            n_docs = len(X)
            n_words = len(self.vocab)
            
            self.word_topic_counts = np.zeros((n_words, self.n_topics))
            self.doc_topic_counts = np.zeros((n_docs, self.n_topics))
            self.topic_counts = np.zeros(self.n_topics)
            self.doc_lengths = np.array([len(doc) for doc in X])
            
            # Randomly assign topics to words
            self.documents = X
            self.word_topics = []
            
            for doc_idx, doc in enumerate(X):
                doc_topics = []
                for word in doc:
                    word_idx = self.vocab[word]
                    topic = random.randint(0, self.n_topics - 1)
                    doc_topics.append((word_idx, topic))
                    
                    # Update counts
                    self.word_topic_counts[word_idx, topic] += 1
                    self.doc_topic_counts[doc_idx, topic] += 1
                    self.topic_counts[topic] += 1
                self.word_topics.append(doc_topics)
    
    def _gibbs_sample(self):
        """Perform one iteration of Gibbs sampling."""
        for doc_idx, doc in enumerate(self.documents):
            for word_pos, (word_idx, old_topic) in enumerate(self.word_topics[doc_idx]):
                # Decrement counts for old topic
                self.word_topic_counts[word_idx, old_topic] -= 1
                self.doc_topic_counts[doc_idx, old_topic] -= 1
                self.topic_counts[old_topic] -= 1
                
                # Calculate topic probabilities
                topic_probs = (
                    (self.word_topic_counts[word_idx, :] + self.beta) *
                    (self.doc_topic_counts[doc_idx, :] + self.alpha) /
                    (self.topic_counts + self.beta * len(self.vocab))
                )
                
                # Sample new topic
                topic_probs = topic_probs / topic_probs.sum()
                new_topic = np.random.choice(self.n_topics, p=topic_probs)
                
                # Update counts for new topic
                self.word_topic_counts[word_idx, new_topic] += 1
                self.doc_topic_counts[doc_idx, new_topic] += 1
                self.topic_counts[new_topic] += 1
                
                # Update topic assignment
                self.word_topics[doc_idx][word_pos] = (word_idx, new_topic)
    
    def fit(self, X: Union[List[List[str]], csr_matrix], feature_names: List[str] = None):
        """
        Fit the LDA model to the given documents.
        
        Args:
            X: Either a list of word lists or a sparse matrix from CountVectorizer
            feature_names: List of feature names (required if X is a sparse matrix)
        """
        if isinstance(X, csr_matrix) and feature_names is None:
            raise ValueError("feature_names must be provided when X is a sparse matrix")
            
        self._initialize_counts(X, feature_names)
        
        for _ in range(self.max_iter):
            self._gibbs_sample()
    
    def get_topic_word_distribution(self) -> np.ndarray:
        """
        Get the topic-word distribution matrix.
        
        Returns:
            np.ndarray: Matrix of shape (n_topics, n_words) containing topic-word probabilities
        """
        return (self.word_topic_counts + self.beta) / (self.topic_counts[:, np.newaxis] + self.beta * len(self.vocab))
    
    def get_document_topic_distribution(self) -> np.ndarray:
        """
        Get the document-topic distribution matrix.
        
        Returns:
            np.ndarray: Matrix of shape (n_docs, n_topics) containing document-topic probabilities
        """
        return (self.doc_topic_counts + self.alpha) / (self.doc_lengths[:, np.newaxis] + self.alpha * self.n_topics)
    
    def get_top_words(self, n_words: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Get the top words for each topic.
        
        Args:
            n_words (int): Number of top words to return per topic
            
        Returns:
            List[List[Tuple[str, float]]]: List of topics, where each topic is a list of (word, probability) tuples
        """
        topic_word_dist = self.get_topic_word_distribution()
        top_words = []
        
        for topic_idx in range(self.n_topics):
            # Get indices of top words for this topic
            top_word_indices = np.argsort(topic_word_dist[topic_idx])[-n_words:][::-1]
            
            # Get words and their probabilities
            topic_words = [
                (self.inv_vocab[idx], topic_word_dist[topic_idx, idx])
                for idx in top_word_indices
            ]
            top_words.append(topic_words)
            
        return top_words
