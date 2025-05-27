import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_data(filename):
    label = LabelEncoder()

    data = pd.read_csv(filename)
    data = data.dropna()
    data['Pop'] = label.fit_transform(data['Pop'])
    data['sex'] = label.fit_transform(data['sex'])
    y = data['age'].to_numpy()
    del data['age']
    X = data.to_numpy()
    return X, y