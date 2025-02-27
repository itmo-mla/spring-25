from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt 
import os 
from scripts import cross_val_score_mine, data_breast_cancer_preprocessing, ploter, my_grid_search, data_housing
import itertools
import warnings
import time  # Добавляем импорт для измерения времени
warnings.filterwarnings("ignore")





class RandomForest:
    def __init__(self,
                 n_estimators:int,
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 sample_size: float,
                 method: Union[DecisionTreeClassifier, DecisionTreeRegressor] = DecisionTreeClassifier,
                 oob_score: bool = False,
                 max_features: float = None):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.sample_size = sample_size
        self.method = method
        self.oob_score_param = oob_score
        self.max_features = max_features
    
    def bootstrap_sample(self, X):
        n_samples = X.shape[0]
        idx_for_sample = np.random.choice(n_samples, size=int(n_samples * self.sample_size), replace=True)
        oob_idx = np.setdiff1d(np.arange(n_samples), idx_for_sample)
        return idx_for_sample, oob_idx
    
    def fit(self, X, y):
        self.trees = []
        self.feature_indices = []
        self.oob_idx = []
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        n_features = X.shape[1]
        if self.max_features is None:
            if self.method == DecisionTreeClassifier:
                self.max_features = "sqrt"
            elif self.method == DecisionTreeRegressor:
                self.max_features = (n_features // 3) / n_features
        
            
        for _ in range(self.n_estimators):
            idx_for_sample, oob_idx = self.bootstrap_sample(X)
            tree = self.method(max_depth=self.max_depth,
                               min_samples_split=self.min_samples_split,
                               min_samples_leaf=self.min_samples_leaf,
                               max_features=self.max_features,
                               ccp_alpha=0.0)
            
            tree.fit(self.X_train[idx_for_sample], self.y_train[idx_for_sample])
            self.trees.append(tree)
            self.oob_idx.append(oob_idx)
        
        if self.oob_score_param:
            self.oob_score_ = self.compute_oob_score()
            
    def compute_oob_score(self):
        n_samples = self.X_train.shape[0]
        predictions = np.zeros((n_samples, self.n_estimators))
        oob_mask = np.zeros((n_samples, self.n_estimators), dtype=bool)
        for i, (tree, oob_indices) in enumerate(zip(self.trees, self.oob_idx)):
            if len(oob_indices) > 0:
                predictions[oob_indices, i] = tree.predict(self.X_train[oob_indices])
                oob_mask[oob_indices, i] = True
        oob_predictions = []
        valid_samples = []
        
        for i in range(n_samples):
            if np.any(oob_mask[i]):
                tree_preds = predictions[i, oob_mask[i]]
                if self.method == DecisionTreeClassifier:
                    oob_predictions.append(np.argmax(np.bincount(tree_preds.astype(int))))
                else:
                    oob_predictions.append(np.mean(tree_preds))
                valid_samples.append(i)
        oob_predictions = np.array(oob_predictions)
        valid_samples = np.array(valid_samples)
        
        if len(valid_samples) == 0:
            return 0.0
        
        if self.method == DecisionTreeClassifier:
            return accuracy_score(self.y_train[valid_samples], oob_predictions)
        else:
            return r2_score(self.y_train[valid_samples], oob_predictions)



class RandomForestRegressor_(RandomForest):
    def __init__(self,
                 n_estimators:int,
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 sample_size: float,
                 oob_score: bool = False,
                 max_features: float = None):
        super().__init__(n_estimators, max_depth, min_samples_split, min_samples_leaf, sample_size,
                        method=DecisionTreeRegressor, oob_score=oob_score, max_features=max_features)
    
    def predict(self, X):
        tree_preds = []
        for i, tree in enumerate(self.trees):
            X_subset = np.array(X)
            tree_preds.append(tree.predict(X_subset))
        tree_preds = np.array(tree_preds).T
        final_preds = []
        for pred in tree_preds:
            final_preds.append(np.mean(pred))
        return np.array(final_preds)
    def score(self, X, y):
        return r2_score(y, self.predict(X))


class RandomForestClassifier_(RandomForest):
    def __init__(self,
                 n_estimators:int,
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 sample_size: float,
                 oob_score: bool = False,
                 max_features: float = None):
        super().__init__(n_estimators, max_depth, min_samples_split, min_samples_leaf, sample_size, 
                         method=DecisionTreeClassifier, oob_score=oob_score, max_features=max_features)
    
    def predict(self, X):
        tree_preds = []
        for i, tree in enumerate(self.trees):
            X_subset = np.array(X)
            tree_preds.append(tree.predict(X_subset))
        tree_preds = np.array(tree_preds).T
        final_preds = []
        for pred in tree_preds:
            final_preds.append(np.argmax(np.bincount(pred.astype(int))))
        return np.array(final_preds)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
        

if __name__ == "__main__":
    X, y = data_breast_cancer_preprocessing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Обучение модели с OOB оценкой
    clf_oob = RandomForestClassifier_(n_estimators=10, max_depth=10, min_samples_split=2, 
                                     min_samples_leaf=1, sample_size=0.6, oob_score=True)

        
    rf_sklearn = RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=2, 
                                       min_samples_leaf=1, oob_score=True)
    
    # Измерение времени обучения
    start_time = time.time()
    clf_oob.fit(X, y)
    my_fit_time = time.time() - start_time
    
    start_time = time.time()
    rf_sklearn.fit(X_train, y_train)
    sklearn_fit_time = time.time() - start_time
    
    print(f"Время обучения (моя реализация): {my_fit_time:.4f} сек")
    print(f"Время обучения (sklearn): {sklearn_fit_time:.4f} сек")
    print(f"Sklearn OOB Score: {rf_sklearn.oob_score_}")
    print(f"OOB Score: {clf_oob.oob_score_}")   
    print("________________________________________________________")

    best_n_estimator, best_max_features, best_sample_size, best_score = my_grid_search(RandomForestClassifier_, X, y)
    clf_oob = RandomForestClassifier_(n_estimators=best_n_estimator,
                                      max_depth=10,
                                      min_samples_split=2, 
                                      min_samples_leaf=1, 
                                      sample_size=best_sample_size, 
                                      max_features=best_max_features, 
                                      oob_score=True)
    
    # Добавляем вызов метода fit перед использованием predict
    clf_oob.fit(X_train, y_train)
        
    rf_sklearn = RandomForestClassifier(n_estimators=best_n_estimator,
                                        max_depth=10,
                                        min_samples_split=2, 
                                        min_samples_leaf=1,
                                        bootstrap=True,
                                        max_samples=best_sample_size,
                                        max_features=best_max_features,
                                        oob_score=True)
    rf_sklearn.fit(X_train, y_train)
    
    # Измерение времени предсказания
    start_time = time.time()
    clf_oob.predict(X_test)
    my_predict_time = time.time() - start_time
    
    start_time = time.time()
    rf_sklearn.predict(X_test)
    sklearn_predict_time = time.time() - start_time
    
    print(f"Время предсказания (моя реализация): {my_predict_time:.4f} сек")
    print(f"Время предсказания (sklearn): {sklearn_predict_time:.4f} сек")
    print(f"Лучшее количество деревьев: {best_n_estimator}")
    print(f"Лучший доля признаков: {best_max_features}") 
    print(f"Лучший размер выборки: {best_sample_size}")
    print(f"Лучшая точность на моей grid search: {best_score}")
    print(f"Результат работы sklearn c этими гиперпараметрами: {rf_sklearn.score(X_test, y_test)}")
    print("________________________________________________________")
    scores_mine = cross_val_score_mine(clf_oob, X, y, cv=5)
    scores_sklearn = cross_val_score_mine(rf_sklearn, X, y, cv=5)
    ploter(scores_sklearn, scores_mine, clf_oob, rf_sklearn.oob_score_)


    #Метрики 
    print(f"Точность Random Forest sklearn: {scores_sklearn.mean()}")
    print(f"Точность Random Forest mine: {scores_mine.mean()}")

    print("________________________________________________________\n")
    print("#####=====RandomForestRegressor=====#####\n")

    X, y = data_housing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    best_n_estimator, best_max_features, best_sample_size, best_score = my_grid_search(RandomForestRegressor_, X, y)
    clf_oob = RandomForestRegressor_(n_estimators=best_n_estimator,
                                      max_depth=10,
                                      min_samples_split=2, 
                                      min_samples_leaf=1, 
                                      sample_size=best_sample_size, 
                                      max_features=best_max_features, 
                                      oob_score=True)
    
    # Измерение времени обучения регрессора
    start_time = time.time()
    clf_oob.fit(X_train, y_train)
    my_fit_time_reg = time.time() - start_time
        
    rf_sklearn = RandomForestRegressor(n_estimators=best_n_estimator,
                                        max_depth=10,
                                        min_samples_split=2, 
                                        min_samples_leaf=1,
                                        bootstrap=True,
                                        max_samples=best_sample_size,
                                        max_features=best_max_features,
                                        oob_score=True)
    
    start_time = time.time()
    rf_sklearn.fit(X_train, y_train)
    sklearn_fit_time_reg = time.time() - start_time
    
    print(f"Время обучения регрессора (моя реализация): {my_fit_time_reg:.4f} сек")
    print(f"Время обучения регрессора (sklearn): {sklearn_fit_time_reg:.4f} сек")
    # Измерение времени предсказания регрессора
    start_time = time.time()
    clf_oob.predict(X_test)
    my_predict_time_reg = time.time() - start_time
    
    start_time = time.time()
    rf_sklearn.predict(X_test)
    sklearn_predict_time_reg = time.time() - start_time
    
    print(f"Время предсказания регрессора (моя реализация): {my_predict_time_reg:.4f} сек")
    print(f"Время предсказания регрессора (sklearn): {sklearn_predict_time_reg:.4f} сек")
    print(f"MSE моя реализация: {mean_squared_error(y_test, clf_oob.predict(X_test))}")
    print(f"MAE моя реализация: {mean_absolute_error(y_test, clf_oob.predict(X_test))}")
    print(f"Sklearn MSE: {mean_squared_error(y_test, rf_sklearn.predict(X_test))}")
    print(f"Sklearn MAE: {mean_absolute_error(y_test, rf_sklearn.predict(X_test))}")
    print("________________________________________________________")
    
    # Добавляем кросс-валидацию для регрессора
    scores_mine_reg = cross_val_score_mine(clf_oob, X, y, cv=5)
    scores_sklearn_reg = cross_val_score_mine(rf_sklearn, X, y, cv=5)

    ploter(scores_sklearn_reg, scores_mine_reg, clf_oob, rf_sklearn.oob_score_)

    # Метрики
    print(f"Средний R² (кросс-валидация) sklearn: {scores_sklearn_reg.mean():.4f}")
    print(f"Средний R² (кросс-валидация) mine: {scores_mine_reg.mean():.4f}")

    