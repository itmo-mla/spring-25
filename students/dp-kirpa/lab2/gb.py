import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.metrics import mean_squared_error

seed = 42
np.random.seed(seed)
random.seed(seed)

"""## Данные"""

data = pd.read_csv("gym_members_exercise_tracking.csv")
data.head()

data.Fat_Percentage.hist()

label_encoder = LabelEncoder()

data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Workout_Type'] = label_encoder.fit_transform(data['Workout_Type'])
data.head()

X, y = data.drop(columns=['Weight (kg)']), data['Weight (kg)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""### Ручной бустинг без подбора весов"""

class MyFirstGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        current_prediction = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = y - current_prediction

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            predictions_update = tree.predict(X)
            current_prediction += self.learning_rate * predictions_update

    def predict(self, X):
        prediction = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)
        return prediction

gb_model = MyFirstGradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)
mean_squared_error(y_test, y_pred)

"""### Ручной бустинг с подбором весов

```
(a(x)*al-y)**2' = 2 * (a(x)*al-y) * a(x) = 0

a(x)**2 * al - a(x) * y = 0

al = a(x) * y / a(x) ** 2
```
"""

class MyGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
        self.alphas = []

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        current_prediction = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = y - current_prediction

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            predictions_update = tree.predict(X)
            current_prediction += self.learning_rate * predictions_update
            self.alphas.append(np.mean(current_prediction * y / (current_prediction * current_prediction)))

    def predict(self, X):
        prediction = np.full(X.shape[0], self.initial_prediction)
        for tree, alpha in zip(self.trees, self.alphas):
            prediction += self.learning_rate * tree.predict(X) * alpha
        return prediction

gb_model = MyGradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)
mean_squared_error(y_test, y_pred)

"""### Библиотечный бустинг"""

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)
y_train_pred = gbr.predict(X_train)
y_test_pred = gbr.predict(X_test)
print("Training MSE:", mean_squared_error(y_train, y_train_pred))
print("Testing MSE:", mean_squared_error(y_test, y_test_pred))