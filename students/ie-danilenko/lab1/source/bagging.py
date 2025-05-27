import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class BaggingClassificator:
    def __init__(self, n_estimators=10, max_samples=1.0, random_state=None):
        self.__models = []
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y):
        n_samples = X.shape[0]
        rand_generator = np.random.RandomState(self.random_state)

        sample_size = int(self.max_samples * n_samples)

        for _ in tqdm(range(self.n_estimators)):
            seed = rand_generator.randint(0, np.iinfo(np.int32).max)
            indices = rand_generator.choice(n_samples, size=sample_size, replace=True)
            features = rand_generator.choice(X.shape[1], round(np.sqrt(X.shape[1])))
            X_boot, y_boot = X[indices][:, features], y[indices]

            tree = DecisionTreeClassifier(random_state=seed)
            tree.fit(X_boot, y_boot)

            X_test = X[~indices][:, features]
            y_test = y[~indices]
            if accuracy_score(tree.predict(X_test), y_test) > 0.8:
                self.__models.append([tree, features])

    def predict(self, X):
        predictions = np.array([tree.predict(X[:, features]) for tree, features in self.__models])

        majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        return majority_votes

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from read import read_data
    from metrics import all_metrics

    X, y = read_data('data/train.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    model = BaggingClassificator(n_estimators=100, max_samples=0.8, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc, rec, prec, f1 = all_metrics(y_test, y_pred)
    print(f"Accuracy: {acc}")
    print(f"Recall: {rec}")
    print(f"Precision: {prec}")
    print(f"f1: {f1}")