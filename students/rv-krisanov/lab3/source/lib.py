import numpy as np


def gaussian(x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> float:
    return (1 / (np.sqrt(2 * np.pi) * x_std)) * np.exp(
        -0.5 * ((x - x_mean) / x_std) ** 2
    )


class GaussianNaiveBayesClassifier:
    def __init__(self): ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        self.y_labels, self.y_counts = np.unique(y, return_counts=True)
        self.y_frequencies = self.y_counts / len(y)  # aprior class probabilities

        self.X_mean = np.array(
            [np.mean(X[y == y_label], axis=0) for y_label in self.y_labels]
        )

        self.X_std = np.array(
            [np.std(X[y == y_label], axis=0) for y_label in self.y_labels]
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        X = X.to_numpy()
        X_mean_deviation = np.array(
            [
                self.pdf(
                    x,
                )
                for x in X
            ]
        )

        posteriors = self.y_frequencies * np.prod(X_mean_deviation, axis=2)
        return np.argmax(posteriors, axis=1)

    def pdf(
        self,
        x: np.array,
    ) -> np.array:
        return gaussian(x, self.X_mean, self.X_std)

