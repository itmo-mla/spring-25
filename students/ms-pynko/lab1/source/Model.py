import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class BaggingClassifierManual:
    def __init__(self, n_estimators=10, max_samples=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.rng_ = np.random.RandomState(self.random_state)
        self.estimators_ = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        # Определение размера выборки для бутстрэпа
        if isinstance(self.max_samples, float):
            sample_size = int(self.max_samples * n_samples)
        else:
            sample_size = self.max_samples

        # Генерация случайных seed для каждого дерева
        seeds = self.rng_.randint(0, np.iinfo(np.int32).max, size=self.n_estimators)

        for seed in seeds:
            # Создание бутстрэп-выборки
            indices = self.rng_.choice(n_samples, size=sample_size, replace=True)
            features = self.rng_.choice(X.shape[1], round(np.sqrt(X.shape[1])))
            X_boot, y_boot = X[indices][:, features], y[indices]

            # Обучение дерева
            tree = DecisionTreeClassifier(random_state=seed)
            tree.fit(X_boot, y_boot)

            # Проверка алгоритма
            X_test, y_test = X[~indices][:, features], y[~indices]
            if accuracy_score(tree.predict(X_test), y_test) > 0.6:
                self.estimators_.append([tree, features])

            if len(self.estimators_) < self.n_estimators:
                good = list(self.estimators_)
                i = 0
                while len(self.estimators_) < self.n_estimators
                    self.estimators_.append(good[i % len(good)])
                    i += 1

    def predict(self, X):
        # Сбор предсказаний всех деревьев
        predictions = np.array([tree.predict(X[:, features]) for tree, features in self.estimators_])
        # Мажоритарное голосование
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        return majority_votes
