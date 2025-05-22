import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from bayes_classifier import NaiveBayesClassifier

data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Обучение собственной реализации
nb_custom = NaiveBayesClassifier()

start_time = time.time()
nb_custom.fit(X_train, y_train)
custom_time = time.time() - start_time
print(f"Time (Custom): {custom_time:.4f}")

y_pred_custom = nb_custom.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print(f"Accuracy (Custom): {accuracy_custom:.4f}")

from sklearn.naive_bayes import MultinomialNB

# Обучение эталонной модели
nb_sklearn = MultinomialNB()

start_time = time.time()
nb_sklearn.fit(X_train_vec, y_train)
sklearn_time = time.time() - start_time
print(f"Time (sklearn): {sklearn_time:.4f}")

y_pred_sklearn = nb_sklearn.predict(X_test_vec)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Accuracy (sklearn): {accuracy_sklearn:.4f}")

def cross_validate_custom_model(X, y, clf, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Обучение модели
        clf.fit(X_train, y_train)

        # Предсказание и оценка точности
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)

# Преобразование данных в массивы numpy
X = data['message'].values
y = data['label'].values

# Кросс-валидация собственной модели
mean_accuracy_custom, std_accuracy_custom = cross_validate_custom_model(X, y, NaiveBayesClassifier(), k=5)
print(f"Mean Accuracy (Custom): {mean_accuracy_custom:.4f} ± {std_accuracy_custom:.4f}")

# Векторизация текста
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(data['message'])

# Кросс-валидация эталонной модели
nb_sklearn = MultinomialNB()
scores_sklearn = cross_val_score(nb_sklearn, X_vec, data['label'], cv=5, scoring='accuracy')

mean_accuracy_sklearn = np.mean(scores_sklearn)
std_accuracy_sklearn = np.std(scores_sklearn)
print(f"Mean Accuracy (sklearn): {mean_accuracy_sklearn:.4f} ± {std_accuracy_sklearn:.4f}")