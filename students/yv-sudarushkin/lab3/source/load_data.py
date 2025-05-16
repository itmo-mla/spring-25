import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data():
    df = pd.read_csv('datasets/Rice-Gonen andJasmine.csv')
    return df


def delete_colum(df:pd.DataFrame):
    df = df.drop(['id'], axis=1)
    return df

# def encod(df: pd.DataFrame):
#     label_encoder = LabelEncoder()
#     df['Class'] = label_encoder.fit_transform(df['Class'])
#     return df


def load_scaled_data():
    # Преобразование категориальных переменных в числовые
    df = load_data()
    df = delete_colum(df)
    # df = encod(df)
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column].astype(str))
    numeric_cols = df.drop('Class', axis=1).select_dtypes(include=[float, int]).columns
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

    return df
