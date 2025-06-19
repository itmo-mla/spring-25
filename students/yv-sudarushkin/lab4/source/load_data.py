import kagglehub
import pandas as pd
import pathlib
import os

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')


def download_data(path=".\\") -> str:
    # Download latest version
    os.environ['KAGGLEHUB_CACHE'] = path

    return kagglehub.dataset_download("imoore/60k-stack-overflow-questions-with-quality-rate")


def _clean_text(text, stop_words, lemmatizer):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)  # удалить HTML
    text = re.sub(r"[^a-zA-Z]", " ", text)  # оставить только буквы
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)


def _preprocess(df, stop_words, lemmatizer):

    df = df[['Title', 'Body', 'Tags']].dropna()
    # --- Объединение названия и тела ---
    df['text'] = df['Title'] + " " + df['Body']
    # --- Предобработка текста ---
    df['clean_text'] = df['text'].apply(_clean_text, args=(stop_words, lemmatizer))

    return df


def load_data(n_samples=None):

    path = download_data()
    df_train = pd.read_csv(pathlib.Path(path) / "train.csv")
    # df_valid = pd.read_csv(pathlib.Path(path) / "valid.csv")
    df_valid = None

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    if n_samples is None:
        df_train = _preprocess (df_train, stop_words, lemmatizer)
    else:
        df_train = _preprocess(df_train.sample(n_samples), stop_words, lemmatizer)
    # df_valid = _preprocess (df_valid, stop_words, lemmatizer)

    return df_train, df_valid
