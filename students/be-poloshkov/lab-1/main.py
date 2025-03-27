from typing import Callable, TypeVar

from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

import time

from bagging import BaggingClassifier as MyBaggingClassifier

random_state = 31

T = TypeVar('T')
def time_elapsed(f: Callable[[...], T], *args, **kwargs) -> tuple[float, T]:
    start = time.time()
    res = f(*args, **kwargs)
    end = time.time()
    return end - start, res

def main():
    data = load_iris()
    X, y = data.data, data.target

    for estimator in (
            DecisionTreeClassifier(max_depth=1, random_state=random_state),
            DecisionTreeClassifier(max_depth=2, random_state=random_state),
    ):
        for n_estimators in (1, 5, 10):
            for label, model in (
                    ('My Bagging', MyBaggingClassifier),
                    ('Sklearn Bagging', BaggingClassifier),
            ):
                bagging = model(estimator=estimator, n_estimators=n_estimators, random_state=random_state)
                time_spent, scores = time_elapsed(cross_val_score, bagging, X, y, scoring='accuracy')

                print(f'{label.upper()}: {estimator}')
                print(f'Количество классификаторов: {n_estimators}')
                print(f'Время работы: {time_spent:.3f}')
                print(f'Кросс-валидация: {scores.mean():.3f}')


if __name__ == '__main__':
    main()