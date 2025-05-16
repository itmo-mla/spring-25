import numpy as np


class GaussianNBClassifier:
    def __init__(self):
        self.class_probs = None
        self.means = None
        self.variances = None
        self.class_labels = None
        self.num_classes = None

    def fit(self, features, labels):
        self.class_labels = np.unique(labels)
        self.num_classes = len(self.class_labels)
        num_features = features.shape[1]

        self.class_probs = np.array([
            np.mean(labels == label) for label in self.class_labels
        ])

        self.means = np.zeros((self.num_classes, num_features))
        self.variances = np.zeros((self.num_classes, num_features))

        for idx, label in enumerate(self.class_labels):
            class_features = features[labels == label]
            self.means[idx] = np.mean(class_features, axis=0)
            self.variances[idx] = np.var(class_features, axis=0, ddof=1) + 1e-9

        return self

    def predict(self, samples):
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples)

        log_class_probs = np.log(self.class_probs)
        predictions = []

        for sample in samples:
            log_likelihoods = -0.5 * np.sum(
                np.log(2 * np.pi * self.variances) +
                (sample - self.means) ** 2 / self.variances,
                axis=1
            )

            posterior_scores = log_class_probs + log_likelihoods
            predicted_class = self.class_labels[np.argmax(posterior_scores)]
            predictions.append(predicted_class)

        return np.array(predictions)
