import numpy as np

from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.word_likelihoods = defaultdict(dict)
        self.classes = None
        self.vocab = None

    def fit(self, X, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.class_priors = {cls: count / len(y) for cls, count in zip(self.classes, counts)}

        for cls in self.classes:
            cls_X = X[y == cls]
            total_words = 0
            word_counts = defaultdict(int)
            for doc in cls_X:
                words = doc.split()
                total_words += len(words)
                for word in words:
                    word_counts[word] += 1

            for word, count in word_counts.items():
                self.word_likelihoods[cls][word] = (count + 1) / (total_words + len(word_counts))

    def predict(self, X):
        predictions = []
        for doc in X:
            log_probs = {}
            for cls in self.classes:
                log_prob = np.log(self.class_priors[cls])
                for word in doc.split():
                    log_prob += np.log(self.word_likelihoods[cls].get(word, 1e-6))
                log_probs[cls] = log_prob
            predictions.append(max(log_probs, key=log_probs.get))
        return predictions
