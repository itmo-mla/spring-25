import pandas as pd

def read_movies(rating_filepath):
    rating = pd.read_csv(rating_filepath)
    return rating