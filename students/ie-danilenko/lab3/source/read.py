import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_heart(filepath):
    encoder = LabelEncoder()

    data = pd.read_csv(filepath, index_col=None)
    data['Sex'] = encoder.fit_transform(data['Sex'])
    data['ChestPainType'] = encoder.fit_transform(data['ChestPainType'])
    data['RestingECG'] = encoder.fit_transform(data['RestingECG'])
    data['ExerciseAngina'] = encoder.fit_transform(data['ExerciseAngina'])
    data['ST_Slope'] = encoder.fit_transform(data['ST_Slope'])
    y = data['HeartDisease'].to_numpy()
    del data['HeartDisease']

    X = data.to_numpy()
    return X, y