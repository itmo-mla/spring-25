import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools

def cross_val_score_mine(clf, X, y, cv=5):
    X = np.array(X)
    y = np.array(y)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, cv)
    scores = []
    for i in range(cv):
        test_idx = fold_indices[i]
        train_idx = np.setdiff1d(indices, test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx] 
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
    return np.array(scores)

def data_breast_cancer_preprocessing(data_path = "/Users/a.konstantinov/Documents/less/spring-25/students/aa-konstantinov/lab1/source/data.csv"):
    breast_cancer = pd.read_csv(data_path)
    breast_cancer = breast_cancer[breast_cancer.columns[1:]]
    breast_cancer = breast_cancer[breast_cancer.columns[:-1]]
    breast_cancer.dropna()
    breast_cancer = breast_cancer.drop_duplicates()
    breast_cancer = breast_cancer.replace({"M": 1, "B": 0})
    X = breast_cancer.drop(columns=["diagnosis"]).to_numpy()
    y = breast_cancer["diagnosis"].to_numpy()
    return X, y

def data_housing(data_path = "/Users/a.konstantinov/Documents/less/spring-25/students/aa-konstantinov/lab1/source/HousingData.csv.xls"):
    housing = pd.read_csv(data_path)
    housing = housing.dropna()
    X = housing.drop(columns=["TAX"]).to_numpy()
    y = housing["TAX"].to_numpy()
    return X, y

def my_grid_search(clf, X, y):
    n_estimators = np.arange(1, 100, 10)
    max_features = np.arange(0.1, 1, 0.1)
    sample_sizes = [0.6, 0.7, 0.8, 0.9, 1.0]
    it = list(itertools.product(n_estimators, max_features, sample_sizes))
    
    scores = []
    for n_estimator, max_features, sample_size in it:
        clf_oob = clf(n_estimators=n_estimator,
                     max_depth=10,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     sample_size=sample_size,
                     max_features=max_features,
                     oob_score=True)
        clf_oob.fit(X, y)
        scores.append(clf_oob.oob_score_)
    best_idx = np.argmax(scores)
    best_params = it[best_idx]
    best_score = scores[best_idx]
    return best_params[0], best_params[1], best_params[2], best_score

    
def ploter(scores_sklearn, scores_mine, clf_oob, sklearn_oob_score):
    plt.figure(figsize=(10, 5))
    plt.plot(scores_sklearn, label="Random Forest sklearn", color="blue")
    plt.plot(scores_mine, label="Random Forest mine", color="red")
    plt.axhline(y=clf_oob.oob_score_, color='green', linestyle='-', label=f"OOB Score: {clf_oob.oob_score_:.3f}")
    plt.axhline(y=sklearn_oob_score, color='yellow', linestyle='-', label=f"Sklearn OOB Score: {sklearn_oob_score:.3f}")
    plt.legend()
    plt.title("Scores of Random Forest")
    plt.xlabel("Номер фолда")
    plt.ylabel("Точность")
    path = os.path.join(os.path.dirname(__file__), "../images/scores_with_oob.png")
    plt.savefig(path)


