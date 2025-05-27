import pandas as pd
import re

def preprocess(text):
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower().split()

def read_texts(filename):
    data = pd.read_csv(filename)
    data = data['Sentences'].apply(preprocess)
    return [' '.join(t) for t in data.to_list()]