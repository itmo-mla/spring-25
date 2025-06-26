import pandas as pd
import numpy as np
import random

seed = 42
np.random.seed(seed)
random.seed(seed)

"""## Данные

https://www.kaggle.com/datasets/gondimalladeepesh/nvidia-documentation-question-and-answer-pairs
"""

data = pd.read_csv('archive (12).zip')
data['full_text'] = data.question + ' ' + data.answer
data.drop(columns=['question', 'answer', 'Unnamed: 0'], inplace=True)
data

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

documents = data.full_text.tolist()[:100]
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(documents)

num_topics = 10
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

for topic_idx, topic_words in enumerate(lda.components_):
    top_words_idx = topic_words.argsort()[-10:][::-1]
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
    print(f"Тема {topic_idx + 1}: {', '.join(top_words)}")

"""# Тематическое моделирование LDA"""

import numpy as np
from collections import Counter
import re
import random

class LDA:
    def __init__(self, num_topics, alpha=0.1, beta=0.1, iterations=1000):
        self.K = num_topics
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.vocab = None
        self.topic_word_counts = None
        self.topic_counts = None
        self.doc_topic_counts = None
        self.topic_assignments = None

    def preprocess(self, documents):
        tokenized_docs = []
        for doc in documents:
            tokens = re.findall(r'\b\w+\b', doc.lower())
            tokenized_docs.append(tokens)

        word_counts = Counter()
        for doc in tokenized_docs:
            word_counts.update(doc)

        self.vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items())}

        docs_as_ids = []
        for doc in tokenized_docs:
            doc_ids = [self.vocab[word] for word in doc if word in self.vocab]
            docs_as_ids.append(doc_ids)

        return docs_as_ids

    def _initialize_counts(self, docs_as_ids):
        D = len(docs_as_ids)
        V = len(self.vocab)

        self.topic_word_counts = np.zeros((self.K, V), dtype=int)
        self.topic_counts = np.zeros(self.K, dtype=int)
        self.doc_topic_counts = np.zeros((D, self.K), dtype=int)

        self.topic_assignments = []

        for d, doc in enumerate(docs_as_ids):
            topics = []
            for word_id in doc:
                topic = random.randint(0, self.K - 1)
                topics.append(topic)

                self.topic_word_counts[topic, word_id] += 1
                self.topic_counts[topic] += 1
                self.doc_topic_counts[d, topic] += 1

            self.topic_assignments.append(topics)

    def _sample_topic(self, d, n, word_id):
        current_topic = self.topic_assignments[d][n]
        self.topic_word_counts[current_topic, word_id] -= 1
        self.topic_counts[current_topic] -= 1
        self.doc_topic_counts[d, current_topic] -= 1

        V = len(self.vocab)

        probs = (self.doc_topic_counts[d] + self.alpha) * \
                (self.topic_word_counts[:, word_id] + self.beta) / \
                (self.topic_counts + V * self.beta)

        probs = probs / np.sum(probs)

        new_topic = np.random.choice(self.K, p=probs)

        self.topic_word_counts[new_topic, word_id] += 1
        self.topic_counts[new_topic] += 1
        self.doc_topic_counts[d, new_topic] += 1

        return new_topic

    def fit(self, documents):
        docs_as_ids = self.preprocess(documents)
        self._initialize_counts(docs_as_ids)

        for iteration in range(self.iterations):
            for d, doc in enumerate(docs_as_ids):
                for n, word_id in enumerate(doc):
                    new_topic = self._sample_topic(d, n, word_id)
                    self.topic_assignments[d][n] = new_topic

            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.iterations}")

    def get_topic_word_distribution(self):
        V = len(self.vocab)
        topic_word_dist = (self.topic_word_counts + self.beta) / \
                          (self.topic_counts[:, np.newaxis] + V * self.beta)
        return topic_word_dist

    def get_document_topic_distribution(self):
        doc_topic_dist = (self.doc_topic_counts + self.alpha) / \
                         (np.sum(self.doc_topic_counts, axis=1)[:, np.newaxis] + self.K * self.alpha)
        return doc_topic_dist

    def get_top_words(self, n=10):
        topic_word_dist = self.get_topic_word_distribution()
        id_to_word = {idx: word for word, idx in self.vocab.items()}

        top_words = []
        for topic_idx in range(self.K):
            top_word_indices = np.argsort(topic_word_dist[topic_idx])[-n:][::-1]
            topic_words = [id_to_word[idx] for idx in top_word_indices]
            top_words.append(topic_words)

        return top_words

    def infer_topics(self, document, max_iterations=100):
        tokens = re.findall(r'\b\w+\b', document.lower())

        doc_ids = [self.vocab[word] for word in tokens if word in self.vocab]

        if not doc_ids:
            return np.ones(self.K) / self.K

        topics = [random.randint(0, self.K - 1) for _ in doc_ids]
        doc_topic_counts = np.zeros(self.K, dtype=int)
        for topic in topics:
            doc_topic_counts[topic] += 1

        for _ in range(max_iterations):
            for n, word_id in enumerate(doc_ids):
                current_topic = topics[n]
                doc_topic_counts[current_topic] -= 1

                V = len(self.vocab)
                probs = (doc_topic_counts + self.alpha) * \
                        (self.topic_word_counts[:, word_id] + self.beta) / \
                        (self.topic_counts + V * self.beta)

                probs = probs / np.sum(probs)

                new_topic = np.random.choice(self.K, p=probs)
                topics[n] = new_topic
                doc_topic_counts[new_topic] += 1

        topic_dist = (doc_topic_counts + self.alpha) / \
                     (len(doc_ids) + self.K * self.alpha)

        return topic_dist

lda = LDA(num_topics=10, iterations=500)
lda.fit(documents)

top_words = lda.get_top_words(n=5)
for idx, words in enumerate(top_words):
    print(f"Topic {idx + 1}: {', '.join(words)}")

doc_topic_dist = lda.get_document_topic_distribution()
for idx, dist in enumerate(doc_topic_dist):
    print(f"Document {idx + 1}: {dist}")

new_doc = "GPUs revolutionized the field of artificial intelligence."
topic_dist = lda.infer_topics(new_doc)
print(f"Topic distribution for new document: {topic_dist}")

import numpy as np
import re
from collections import Counter
from itertools import combinations
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def calculate_coherence(documents, topic_words, measure='c_v'):
    tokenized_docs = []
    for doc in documents:
        tokens = re.findall(r'\b\w+\b', doc.lower())
        tokenized_docs.append(tokens)

    word_counts = Counter()
    for doc in tokenized_docs:
        word_counts.update(doc)

    word_doc_freq = {}
    for word in word_counts:
        word_doc_freq[word] = sum(1 for doc in tokenized_docs if word in doc)

    co_occur = {}
    window_size = 10

    for doc in tokenized_docs:
        for i in range(len(doc)):
            for j in range(i+1, min(i+window_size, len(doc))):
                word_pair = tuple(sorted([doc[i], doc[j]]))
                co_occur[word_pair] = co_occur.get(word_pair, 0) + 1

    topic_coherences = []

    for topic in topic_words:
        topic_coherence = 0

        if measure == 'c_v':
            word_pairs = list(combinations(topic, 2))
            scores = []

            for word_i, word_j in word_pairs:
                pair = tuple(sorted([word_i, word_j]))

                co_occur_count = co_occur.get(pair, 0)

                count_i = word_counts.get(word_i, 0)
                count_j = word_counts.get(word_j, 0)

                if co_occur_count > 0:
                    pmi = np.log(co_occur_count * len(tokenized_docs) / (count_i * count_j))
                    npmi = pmi / -np.log(co_occur_count / len(tokenized_docs))
                    scores.append(npmi)
                else:
                    scores.append(-1)

            if scores:
                topic_coherence = np.mean(scores)

        elif measure == 'u_mass':
            word_pairs = list(combinations(topic, 2))
            scores = []

            for word_i, word_j in word_pairs:
                pair = tuple(sorted([word_i, word_j]))

                co_doc_freq = sum(1 for doc in tokenized_docs if word_i in doc and word_j in doc)

                doc_freq_j = word_doc_freq.get(word_j, 0)

                if doc_freq_j > 0 and co_doc_freq > 0:
                    score = np.log((co_doc_freq + 1) / doc_freq_j)
                    scores.append(score)
                else:
                    scores.append(float('-inf'))

            if scores:
                topic_coherence = np.mean(scores)

        elif measure == 'c_uci':
            word_pairs = list(combinations(topic, 2))
            scores = []

            for word_i, word_j in word_pairs:
                pair = tuple(sorted([word_i, word_j]))

                co_occur_count = co_occur.get(pair, 0)

                count_i = word_counts.get(word_i, 0)
                count_j = word_counts.get(word_j, 0)

                if co_occur_count > 0:
                    pmi = np.log(co_occur_count * len(tokenized_docs) / (count_i * count_j))
                    scores.append(pmi)
                else:
                    scores.append(float('-inf'))

            if scores:
                topic_coherence = np.mean(scores)

        topic_coherences.append(topic_coherence)

    return np.mean(topic_coherences)

def compare_lda_coherence(documents, num_topics=3, top_n=10):
    custom_lda = LDA(num_topics=num_topics, iterations=500)
    custom_lda.fit(documents)
    custom_top_words = custom_lda.get_top_words(n=top_n)

    # Run scikit-learn LDA
    vectorizer = CountVectorizer(lowercase=True, token_pattern=r'\b\w+\b')
    X = vectorizer.fit_transform(documents)

    sklearn_lda = LatentDirichletAllocation(
        n_components=num_topics,
        max_iter=500,
        learning_method='online',
        random_state=42
    )

    sklearn_lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    sklearn_top_words = []

    for topic_idx, topic in enumerate(sklearn_lda.components_):
        top_features_idx = topic.argsort()[:-top_n-1:-1]
        top_features = [feature_names[i] for i in top_features_idx]
        sklearn_top_words.append(top_features)

    measures = ['c_v', 'u_mass', 'c_uci']
    results = {}

    for measure in measures:
        custom_coherence = calculate_coherence(documents, custom_top_words, measure=measure)
        sklearn_coherence = calculate_coherence(documents, sklearn_top_words, measure=measure)

        results[measure] = {
            'custom': custom_coherence,
            'sklearn': sklearn_coherence
        }

    print("\n============= COHERENCE COMPARISON =============")
    print("\nCustom LDA Implementation:")
    for i, words in enumerate(custom_top_words):
        print(f"Topic {i+1}: {', '.join(words)}")

    print("\nScikit-learn LDA Implementation:")
    for i, words in enumerate(sklearn_top_words):
        print(f"Topic {i+1}: {', '.join(words)}")

    print("\nCoherence Scores:")
    for measure in measures:
        print(f"\n{measure.upper()} Coherence (higher is better):")
        print(f"Custom LDA: {results[measure]['custom']:.4f}")
        print(f"Scikit-learn LDA: {results[measure]['sklearn']:.4f}")

        if results[measure]['custom'] > results[measure]['sklearn']:
            print(f"Result: Custom LDA has higher {measure} coherence")
        elif results[measure]['custom'] < results[measure]['sklearn']:
            print(f"Result: Scikit-learn LDA has higher {measure} coherence")
        else:
            print(f"Result: Both models have equal {measure} coherence")

    return results, custom_top_words, sklearn_top_words


results, custom_topics, sklearn_topics = compare_lda_coherence(documents, num_topics=2, top_n=5)