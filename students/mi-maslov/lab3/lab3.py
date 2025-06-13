import numpy as np
from sklearn.naive_bayes import GaussianNB
import time

class NaiveBayesClassifier:

    def fit(self, X, y):
        self.y_classes, y_counts = np.unique(y, return_counts=True)
        self.phi_y = 1.0 * y_counts / y_counts.sum()

        self.mean = np.zeros((len(self.y_classes), X.shape[1]))
        self.std = np.zeros((len(self.y_classes), X.shape[1]))

        for i, y_class in enumerate(self.y_classes):
            X_class = X[y == y_class]
            self.mean[i, :] = X_class.mean(axis=0)
            self.std[i, :] = X_class.std(axis=0)

        return self

    def gaussian_pdf(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def compute_prob(self, x, y_class_idx):
        Pxy = 1
        for j in range(len(x)):
            mean = self.mean[y_class_idx, j]
            std = self.std[y_class_idx, j]
            if std == 0:
                std = 1e-9
            Pxy *= self.gaussian_pdf(x[j], mean, std)
        return Pxy * self.phi_y[y_class_idx]

    def predict(self, X):
        return np.apply_along_axis(lambda x: self.compute_probs(x), 1, X)

    def compute_probs(self, x):
        probs = np.array([self.compute_prob(x, y_class_idx) for y_class_idx in range(len(self.y_classes))])
        return self.y_classes[np.argmax(probs)]

    def evaluate(self, X, y):
        return (self.predict(X) == y).mean()


from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

nb = NaiveBayesClassifier().fit(X_train, y_train)
nb.evaluate(X_test, y_test)

start_time = time.time()
nb = NaiveBayesClassifier().fit(X_train, y_train)
train_time_custom = time.time() - start_time

start_time = time.time()
accuracy_custom = nb.evaluate(X_test, y_test)
predict_time_custom = time.time() - start_time

print(f"Custom Naive Bayes Accuracy: {accuracy_custom:.4f}")
print(f"Custom Training Time: {train_time_custom:.6f} seconds")
print(f"Custom Prediction Time: {predict_time_custom:.6f} seconds")

print(f"\n{'^0w0^'*10}\n")
start_time = time.time()
nb_sklearn = GaussianNB()
nb_sklearn.fit(X_train, y_train)
train_time_sklearn = time.time() - start_time

start_time = time.time()
accuracy_sklearn = nb_sklearn.score(X_test, y_test)
predict_time_sklearn = time.time() - start_time

print(f"Scikit-learn Naive Bayes Accuracy: {accuracy_sklearn:.4f}")
print(f"Scikit-learn Training Time: {train_time_sklearn:.6f} seconds")
print(f"Scikit-learn Prediction Time: {predict_time_sklearn:.6f} seconds")

k_folds=5

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
accuracies = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    nb.fit(X_train, y_train)

    accuracy = nb.evaluate(X_val, y_val)
    print(accuracy)
    accuracies.append(accuracy)

print(np.mean(accuracies), np.std(accuracies))