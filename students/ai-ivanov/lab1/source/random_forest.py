import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_features: int | None = None,
        random_state: int | None = None,
        bootstrap_size: float = 1.0,
        min_oob_score: float = 0.5,
    ):
        """
        Parameters:
        -----------
        n_estimators : int, default=100
            Количество деревьев в лесу
        max_features : int, optional
            Количество признаков для рассмотрения при поиске лучшего разбиения.
            Если None, используется sqrt(n_features) для классификации
        random_state : int, optional
            Состояние генератора случайных чисел
        bootstrap_size : float, default=1.0
            Размер бутстрэп-выборки как доля от исходного набора данных
        min_oob_score : float, default=0.5
            Минимальный порог качества на OOB выборке для включения дерева в ансамбль
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.bootstrap_size = bootstrap_size
        self.min_oob_score = min_oob_score
        self.trees: list[DecisionTreeClassifier] = []
        self.oob_indices: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """
        Обучение случайного леса

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Обучающие данные
        y : array-like of shape (n_samples,)
            Целевые метки классов
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        # Если max_features не задан, используем sqrt(n_features)
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))

        # Функция для создания и обучения одного дерева
        def _fit_single_tree(i, X, y, n_samples):
            # Bootstrap выборка
            sample_size = int(n_samples * self.bootstrap_size)
            bootstrap_indices = rng.randint(0, n_samples, sample_size)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            # Получаем OOB индексы (те, что не попали в бутстрэп)
            oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(bootstrap_indices))

            # Создаем и обучаем дерево
            tree = DecisionTreeClassifier(
                max_features=self.max_features,
                random_state=self.random_state + i if self.random_state is not None else None,
            )
            tree = tree.fit(X_bootstrap, y_bootstrap)

            # Оцениваем качество на OOB выборке
            if len(oob_indices) > 0:
                oob_predictions = tree.predict(X[oob_indices])
                oob_score = np.mean(oob_predictions == y[oob_indices])
            else:
                oob_score = 0.0

            return tree, oob_indices, oob_score

        # Параллельное обучение деревьев
        results = Parallel(n_jobs=-1)(
            delayed(_fit_single_tree)(i, X, y, n_samples)
            for i in range(self.n_estimators)
        )

        # Фильтруем деревья по качеству на OOB выборке
        self.trees = []
        self.oob_indices = []
        for tree, oob_indices, oob_score in results:
            if oob_score >= self.min_oob_score:
                self.trees.append(tree)
                self.oob_indices.append(oob_indices)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание классов для X

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Тестовые данные

        Returns:
        --------
        y : array-like of shape (n_samples,)
            Предсказанные метки классов
        """
        if not self.trees:
            raise ValueError("No trees in the forest. Try lowering min_oob_score threshold.")

        # Получаем предсказания от всех деревьев
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Используем голосование большинством для итогового предсказания
        # axis=0 для голосования по всем деревьям
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )
