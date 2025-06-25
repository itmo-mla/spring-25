import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
import time

class ClassicBoosting:
    def __init__(self, n_estimators=10, base_estimator=None):
        self.n_estimators = n_estimators
        # Используем дерево решений как базовый алгоритм
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier(max_depth=1)
        self.models = []
        self.alphas = []
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        # Инициализация весов
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # Копируем базовый алгоритм
            model = self.base_estimator.__class__(**self.base_estimator.get_params())
            
            # Обучаем модель с учетом весов
            model.fit(X, y, sample_weight=weights)
            
            # Предсказания
            predictions = model.predict(X)
            
            # Вычисляем ошибку
            error = np.sum(weights * (predictions != y)) / np.sum(weights)
            
            # Вычисляем вес модели (alpha)
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))
            
            # Обновляем веса объектов
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)  # Нормализация
            
            # Сохраняем модель и её вес
            self.models.append(model)
            self.alphas.append(alpha)
    
    def predict(self, X):
        # Суммируем взвешенные предсказания
        predictions = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            predictions += alpha * model.predict(X)
        return np.sign(predictions)

# Пример использования
from sklearn.datasets import load_iris

# Загрузка данных
data = load_iris()
X, y = data.data, data.target
print(y)
# Преобразуем в задачу бинарной классификации (класс 0 против остальных)
y = np.where(y == 2, 1, -1)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
start_time = time.time()
boosting = ClassicBoosting(n_estimators=10)
boosting.fit(X_train, y_train)
training_time = time.time() - start_time

# Предсказание
y_pred = boosting.predict(X_test)

# Оценка качества
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Training time: {training_time:.4f} seconds")


print("\n=== Библиотечный AdaBoost ===")
start_time_lib = time.time()
lib_boosting = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=10,
    random_state=42
)
lib_boosting.fit(X_train, y_train)
lib_training_time = time.time() - start_time_lib

# Предсказание и точность
y_pred_lib = lib_boosting.predict(X_test)
lib_accuracy = accuracy_score(y_test, y_pred_lib)

print(f"Accuracy: {lib_accuracy:.4f}")
print(f"Training time: {lib_training_time:.4f} seconds")
