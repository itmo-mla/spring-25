import pandas as pd

def read_data(filename):
    data = pd.read_csv(filename)
    y = data['price_range'].to_numpy()
    del data['price_range']
    X = data.to_numpy()
    return X, y