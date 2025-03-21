from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

from Model import BaggingClassifierManual

# Загрузка данных
data = pd.read_csv('glass.csv')
X, y = data.iloc[:, :-1], data.iloc[:, -1]
y = pd.Series(LabelEncoder().fit_transform(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train,), np.array(y_test)


def time_of_method(func):
    def wrapper():
        start = time.time()
        func()
        end = time.time()
        delta = end - start
        print(f'{delta}')
    return wrapper


@time_of_method
def hand_method():
    # Обучение бэггинга
    bagging = BaggingClassifierManual(n_estimators=100, max_samples=0.8, random_state=42)
    bagging.fit(X_train, y_train)
    y_pred = bagging.predict(X_test)
    print(f"Accuracy hand method: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall hand method: {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Precession hand method: {precision_score(y_test, y_pred, average='macro'):.4f}")


@time_of_method
def tree_method():
    # Сравнение с одним деревом
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    print(f"Single tree accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
    print(f"Single tree recall: {recall_score(y_test, y_pred_tree, average='macro'):.4f}")
    print(f"Single tree precession: {precision_score(y_test, y_pred_tree, average='macro'):.4f}")


@time_of_method
def sklearn_method():
    model = RandomForestClassifier(n_estimators=100, max_samples=0.8, random_state=42)
    model.fit(X_train, y_train)
    y_pred_model = model.predict(X_test)
    print(f"Sklearn accuracy: {accuracy_score(y_test, y_pred_model):.4f}")
    print(f"Sklearn recall: {recall_score(y_test, y_pred_model, average='macro'):.4f}")
    print(f"Sklearn precession: {precision_score(y_test, y_pred_model, average='macro'):.4f}")


hand_method()
tree_method()
sklearn_method()
