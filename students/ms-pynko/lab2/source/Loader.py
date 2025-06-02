import pandas as pd


class Loader:
    def __init__(self):
        data = pd.read_csv('bikes_rent.csv')

        data = data.dropna()

        features = data.drop(columns=['cnt'], axis=1)
        target = data['cnt']

        self.X, self.y = features, target
