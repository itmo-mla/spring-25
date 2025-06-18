
import numpy as np
from collections import Counter
import re
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.special import gammaln
import time
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class LDA:
    def __init__(self,
                 n_topics: int,
                 alpha: float = 0.1,
                 beta: float = 0.01,
                 max_iterations: int = 50,
                 tolerance: float = 1e-6,
                 random_state: Optional[int] = None):

        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        if random_state is not None:
            np.random.seed(random_state)

    def preprocess_text(self, documents: List[str]) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
        preprocessed_docs = []
        for doc in documents:
            doc = doc.lower()
            doc = re.sub(r'[^\w\s]', '', doc)
            tokens = doc.split()
            preprocessed_docs.append(tokens)

        word_counts = Counter()
        for doc in preprocessed_docs:
            word_counts.update(doc)

        word_to_id = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
        id_to_word = {idx: word for word, idx in word_to_id.items()}

        corpus = [[word_to_id[word] for word in doc if word in word_to_id]
                  for doc in preprocessed_docs]

        return corpus, word_to_id, id_to_word

    def initialize_parameters(self, corpus: List[List[int]], vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.n_docs = len(corpus)

        # Document-term matrix
        self.n_dw = np.zeros((self.n_docs, self.vocab_size))
        for d, doc in enumerate(corpus):
            for word_id in doc:
                self.n_dw[d, word_id] += 1

        # Initialize Phi (word probabilities for each topic)
        self.phi = np.random.dirichlet(
            np.ones(self.vocab_size) * self.beta,
            size=self.n_topics
        )

        # Initialize Theta (topic probabilities for each document)
        self.theta = np.random.dirichlet(
            np.ones(self.n_topics) * self.alpha,
            size=self.n_docs
        )

        # Prepare arrays for E-step counts
        self.n_wt = np.zeros((self.vocab_size, self.n_topics))
        self.n_td = np.zeros((self.n_topics, self.n_docs))

    def e_step(self) -> None:
        # Reset count matrices
        self.n_wt.fill(0)
        self.n_td.fill(0)

        # For each document and word, calculate p(t|d,w)
        for d in range(self.n_docs):
            for w in range(self.vocab_size):
                if self.n_dw[d, w] > 0:
                    # Calculate p(t|d,w) for all topics
                    p_tdw = self.phi[:, w] * self.theta[d, :]
                    p_tdw = p_tdw / np.sum(p_tdw) if np.sum(p_tdw) > 0 else np.ones(self.n_topics) / self.n_topics

                    # Update counts
                    self.n_wt[w, :] += self.n_dw[d, w] * p_tdw
                    self.n_td[:, d] += self.n_dw[d, w] * p_tdw

    def m_step(self) -> None:
        # Update phi (word probabilities for each topic) with beta regularization
        for t in range(self.n_topics):
            self.phi[t, :] = (self.n_wt[:, t] + self.beta - 1)
            self.phi[t, :] = np.maximum(0, self.phi[t, :])
            self.phi[t, :] /= np.sum(self.phi[t, :])

        # Update theta (topic probabilities for each document) with alpha regularization
        for d in range(self.n_docs):
            self.theta[d, :] = (self.n_td[:, d] + self.alpha - 1)
            self.theta[d, :] = np.maximum(0, self.theta[d, :])
            self.theta[d, :] /= np.sum(self.theta[d, :])

    def log_likelihood(self) -> float:
        ll = 0
        for d in range(self.n_docs):
            for w in range(self.vocab_size):
                if self.n_dw[d, w] > 0:
                    p_wd = np.sum(self.phi[:, w] * self.theta[d, :])
                    ll += self.n_dw[d, w] * np.log(p_wd + 1e-10)
        return ll

    def fit(self, corpus: List[List[int]], vocab_size: int) -> 'LDA':
        self.initialize_parameters(corpus, vocab_size)

        prev_ll = -np.inf
        converged = False

        for iteration in range(self.max_iterations):
            # Perform E-step
            self.e_step()

            # Perform M-step
            self.m_step()

            # Check convergence
            ll = self.log_likelihood()
            ll_change = abs(ll - prev_ll)

            print(f"Iteration {iteration+1}/{self.max_iterations}, Log-Likelihood: {ll:.2f}, Change: {ll_change:.4f}")

            if ll_change < self.tolerance:
                print(f"Converged at iteration {iteration+1}")
                converged = True
                break

            prev_ll = ll

        if not converged:
            print(f"Reached maximum iterations without convergence.")

        return self

    def get_topic_words(self, id_to_word: Dict[int, str], n_words: int = 10) -> List[List[str]]:
        topic_words = []
        for t in range(self.n_topics):
            top_word_indices = np.argsort(-self.phi[t, :])[:n_words]
            topic_words.append([id_to_word[idx] for idx in top_word_indices])

        return topic_words

    def get_document_topics(self, n_topics: int = None) -> List[List[Tuple[int, float]]]:
        if n_topics is None:
            n_topics = self.n_topics

        document_topics = []
        for d in range(self.n_docs):
            topic_probs = [(t, self.theta[d, t]) for t in range(self.n_topics)]
            topic_probs.sort(key=lambda x: x[1], reverse=True)
            document_topics.append(topic_probs[:n_topics])

        return document_topics

    def perplexity(self, corpus: List[List[int]]) -> float:
        n_words = 0
        log_prob = 0

        # Create document-term matrix for the test corpus
        n_dw_test = np.zeros((len(corpus), self.vocab_size))
        for d, doc in enumerate(corpus):
            for word_id in doc:
                if word_id < self.vocab_size:  # Make sure word is in vocabulary
                    n_dw_test[d, word_id] += 1
                    n_words += 1

        # Calculate probability of each word in each document
        for d, doc in enumerate(corpus):
            doc_theta = np.ones(self.n_topics) * self.alpha
            for _ in range(5):
                for word_id in doc:
                    if word_id < self.vocab_size:
                        p_tdw = self.phi[:, word_id] * doc_theta
                        p_tdw = p_tdw / np.sum(p_tdw) if np.sum(p_tdw) > 0 else np.ones(self.n_topics) / self.n_topics
                        doc_theta += p_tdw

            doc_theta /= np.sum(doc_theta)

            # Calculate log probability
            for w in range(self.vocab_size):
                if n_dw_test[d, w] > 0:
                    p_wd = np.sum(self.phi[:, w] * doc_theta)
                    log_prob += n_dw_test[d, w] * np.log(p_wd + 1e-10)

        # Calculate perplexity
        return np.exp(-log_prob / max(1, n_words))

    def visualize_topics(self, id_to_word: Dict[int, str], n_words: int = 10, figsize: Tuple[int, int] = (15, 10)) -> None:
        topic_words = self.get_topic_words(id_to_word, n_words)

        fig, axes = plt.subplots(int(np.ceil(self.n_topics / 3)), 3, figsize=figsize)
        axes = axes.flatten()

        for t, (ax, words) in enumerate(zip(axes, topic_words)):
            word_probs = [(word, self.phi[t, id]) for word, id in
                          [(word, list(id_to_word.values()).index(word)) for word in words]]
            word_probs.sort(key=lambda x: x[1], reverse=True)

            words = [word for word, _ in word_probs]
            probs = [prob for _, prob in word_probs]

            ax.barh(np.arange(len(words)), probs, align='center')
            ax.set_yticks(np.arange(len(words)))
            ax.set_yticklabels(words)
            ax.set_title(f'Topic {t+1}')
            ax.set_xlabel('Probability')

        plt.tight_layout()
        plt.show()


class LDAComparison:

    def __init__(self, n_topics=3, alpha=0.1, beta=0.01, max_iter=500, random_state=None):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = random_state

        # Initialize models
        self.custom_lda = LDA(
            n_topics=n_topics,
            alpha=alpha,
            beta=beta,
            max_iterations=max_iter,
            random_state=random_state
        )

        # For sklearn's LDA, we need to convert the parameters to match
        # sklearn uses document_topic_prior (alpha) and topic_word_prior (beta)
        self.sklearn_lda = LatentDirichletAllocation(
            n_components=n_topics,
            doc_topic_prior=alpha,
            topic_word_prior=beta,
            max_iter=max_iter,
            random_state=random_state
        )

    def preprocess_documents(self, documents):
        # First, preprocess with custom tokenizer to get the tokenized corpus
        corpus, word_to_id, id_to_word = self.custom_lda.preprocess_text(documents)

        # Create preprocessed documents as strings for sklearn
        preprocessed_docs = []
        for doc in documents:
            doc = doc.lower()
            doc = re.sub(r'[^\w\s]', '', doc)
            preprocessed_docs.append(doc)

        # Create a vectorizer with the same tokenization pattern
        vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r'\b\w+\b',  # Simple word tokenization like in custom preprocessing
        )

        return preprocessed_docs, corpus, word_to_id, id_to_word, vectorizer

    def compare_performance(self, documents):

        results = {}

        # Preprocess documents
        preprocessed_docs, corpus, word_to_id, id_to_word, vectorizer = self.preprocess_documents(documents)
        vocab_size = len(word_to_id)

        # Create document-term matrix for sklearn
        X = vectorizer.fit_transform(preprocessed_docs)
        feature_names = vectorizer.get_feature_names_out()

        # Train custom LDA
        custom_start_time = time.time()
        self.custom_lda.fit(corpus, vocab_size)
        custom_time = time.time() - custom_start_time
        results['custom_time'] = custom_time
        results['custom_perplexity'] = self.custom_lda.perplexity(corpus)

        # Train sklearn LDA
        sklearn_start_time = time.time()
        self.sklearn_lda.fit(X)
        sklearn_time = time.time() - sklearn_start_time
        results['sklearn_time'] = sklearn_time
        results['sklearn_perplexity'] = self.sklearn_lda.perplexity(X)

        # Get top words for each topic (for quality comparison)
        custom_topics = self.custom_lda.get_topic_words(id_to_word)

        sklearn_topics = []
        for topic_idx, topic in enumerate(self.sklearn_lda.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            sklearn_topics.append(top_words)

        results['custom_topics'] = custom_topics
        results['sklearn_topics'] = sklearn_topics

        # Calculate speedup
        results['speedup'] = custom_time / sklearn_time if sklearn_time > 0 else float('inf')

        return results

    def compare_topic_coherence(self, custom_topics, sklearn_topics):
        # This is a simple topic similarity measure
        # For each custom topic, find the most similar sklearn topic
        max_similarities = []

        for custom_topic in custom_topics:
            similarities = []
            for sklearn_topic in sklearn_topics:
                # Count words in common
                common_words = set(custom_topic).intersection(set(sklearn_topic))
                similarity = len(common_words) / len(custom_topic)
                similarities.append(similarity)

            max_similarities.append(max(similarities))

        # Average of best matches
        return sum(max_similarities) / len(max_similarities)

    def visualize_comparison(self, results):
        # 1. Execution time comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        times = [results['custom_time'], results['sklearn_time']]
        plt.bar(['Custom LDA', 'Sklearn LDA'], times)
        plt.title('Execution Time Comparison')
        plt.ylabel('Time (seconds)')

        # 2. Perplexity comparison (lower is better)
        plt.subplot(1, 2, 2)
        perplexities = [results['custom_perplexity'], results['sklearn_perplexity']]
        plt.bar(['Custom LDA', 'Sklearn LDA'], perplexities)
        plt.title('Perplexity Comparison (lower is better)')
        plt.ylabel('Perplexity')

        plt.tight_layout()
        plt.show()

        # 3. Topic similarity visualization
        topic_similarity = self.compare_topic_coherence(
            results['custom_topics'],
            results['sklearn_topics']
        )

        print(f"Topic similarity between models: {topic_similarity:.2f} (higher is better)")

        # 4. Compare top words for each topic
        print("\nTop words comparison:")
        for t in range(self.n_topics):
            print(f"\nTopic {t+1}:")
            print(f"Custom LDA: {', '.join(results['custom_topics'][t])}")
            print(f"Sklearn LDA: {', '.join(results['sklearn_topics'][t])}")


import pandas as pd

df = pd.read_csv("abcnews-date-text.csv")

documents = df["headline_text"].to_numpy()[100:201]

lda_comparison = LDAComparison(n_topics=4, random_state=42)
results = lda_comparison.compare_performance(documents)

print(f"\nExecution time comparison:")
print(f"Custom LDA: {results['custom_time']:.4f} seconds")
print(f"Sklearn LDA: {results['sklearn_time']:.4f} seconds")
print(f"Speedup (sklearn vs custom): {1/results['speedup']:.2f}x")

print(f"\nPerplexity comparison (lower is better):")
print(f"Custom LDA: {results['custom_perplexity']:.2f}")
print(f"Sklearn LDA: {results['sklearn_perplexity']:.2f}")

lda_comparison.visualize_comparison(results)