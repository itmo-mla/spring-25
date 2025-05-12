import numpy as np

class GaussianNBClassifier:
    def train(self, features, labels):
        self.class_labels = np.unique(labels)
        self.num_classes = len(self.class_labels)
        self.num_features = features.shape[1]

        self.class_probs = np.array([
            np.mean(labels == label) for label in self.class_labels
        ])

        self.means = np.zeros((self.num_classes, self.num_features))
        self.variances = np.zeros((self.num_classes, self.num_features))

        for idx, label in enumerate(self.class_labels):
            class_features = features[labels == label]
            self.means[idx] = np.mean(class_features, axis=0)
            self.variances[idx] = np.var(class_features, axis=0) + 1e-9  # для численной стабильности

    def classify(self, samples):
        predictions = []
        log_class_probs = np.log(self.class_probs)

        for sample in samples:
            log_likelihoods = []

            for idx in range(self.num_classes):
                mean = self.means[idx]
                var = self.variances[idx]
                log_prob = -0.5 * np.sum(np.log(2 * np.pi * var))
                log_prob -= 0.5 * np.sum((sample - mean) ** 2 / var)
                log_likelihoods.append(log_prob)

            posterior_scores = log_class_probs + log_likelihoods
            predicted_class = self.class_labels[np.argmax(posterior_scores)]
            predictions.append(predicted_class)

        return np.array(predictions)
