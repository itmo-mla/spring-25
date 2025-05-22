import kagglehub
import pandas as pd
import pathlib
import os


def download_data(path=".\\") -> str:
    # Download latest version
    os.environ['KAGGLEHUB_CACHE'] = path

    return kagglehub.dataset_download("CooperUnion/anime-recommendations-database")


def load_data():

    path = download_data()
    df = pd.read_csv(pathlib.Path(path) / "rating.csv")
    names = pd.read_csv(pathlib.Path(path) / "anime.csv")
    return df, names
