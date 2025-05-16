import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler


def load_house() -> tuple[np.ndarray, np.ndarray]:
    """

    :return: X(20640, 8), y (20640,)
    """
    data = fetch_california_housing()
    X, y = data.data, data.target
    return np.array(X), np.array(y)


def load_scaled_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = load_house()

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Масштабируем целевую переменную
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    return X_scaled, y_scaled
