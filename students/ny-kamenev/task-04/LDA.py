import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1, load_files
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import CoherenceModel
from tqdm import tqdm
from time import time
from sklearn.decomposition import LatentDirichletAllocation
import os
import requests
import tarfile
import zipfile
from io import BytesIO
from tabulate import tabulate
import pandas as pd

class LDAModel:

    def __init__(self, n_topics: int,
                 n_iter: int = 50,
                 alpha: float = 0.1,
                 beta: float = 0.01,
                 random_state: int = 42):
        self.K = n_topics
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X):
        X = csr_matrix(X, dtype=np.float64)
        D, V = X.shape

        self.phi = np.random.dirichlet(np.ones(V), self.K)
        self.theta = np.random.dirichlet(np.ones(self.K), D)

        for it in tqdm(range(self.n_iter), desc="LDA"):
            n_wt = np.zeros((self.K, V))
            n_td = np.zeros((D, self.K))

            rows, cols = X.nonzero()
            data = X.data
            for idx in range(len(data)):
                d, w, cnt = rows[idx], cols[idx], data[idx]

                p = self.phi[:, w] * self.theta[d]
                p /= p.sum()

                n_wt[:, w] += cnt * p
                n_td[d] += cnt * p

            self.phi = n_wt + self.beta
            self.phi /= self.phi.sum(axis=1, keepdims=True)
            
            self.theta = n_td + self.alpha
            self.theta /= self.theta.sum(axis=1, keepdims=True)
        
        return self
    
    def transform(self, X):
        if not hasattr(self, "phi"):
            raise RuntimeError("Model must be fitted")
            
        X = csr_matrix(X, dtype=np.float64)
        D, V = X.shape
        theta_new = np.random.dirichlet(np.ones(self.K), D)

        for _ in range(20):
            n_td = np.zeros_like(theta_new)
            rows, cols = X.nonzero()
            data = X.data
            for idx in range(len(data)):
                d, w, cnt = rows[idx], cols[idx], data[idx]
                p = self.phi[:, w] * theta_new[d]
                p /= p.sum()
                n_td[d] += cnt * p
            theta_new = n_td + self.alpha
            theta_new /= theta_new.sum(axis=1, keepdims=True)
        return theta_new
    
    def get_topics(self, feature_names, n_top_words=10):
        topics = []
        for k in range(self.K):
            top_idx = self.phi[k].argsort()[::-1][:n_top_words]
            topics.append([feature_names[i] for i in top_idx])
        return topics

def download_dataset(url, extract_path):
    if not os.path.exists(extract_path):
        print(f"Downloading dataset to {extract_path}...")
        response = requests.get(url)

        if url.endswith('.zip'):
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(extract_path)
        elif url.endswith('.tar.gz') or url.endswith('.tgz'):
            with tarfile.open(fileobj=BytesIO(response.content), mode='r:gz') as tar:
                tar.extractall(path=extract_path)
        else:
            raise ValueError(f"Unsupported format for: {url}")

    for root, dirs, files in os.walk(extract_path):
        if 'bbc' in dirs:
            return os.path.join(root, 'bbc')
        elif any(f.endswith('.txt') for f in files):
            return root
            
    return extract_path

def load_dataset(dataset_name):
    print(f"\nLoading {dataset_name} dataset...")
    
    if dataset_name == '20newsgroups':
        dataset = fetch_20newsgroups(
            subset='train',
            remove=('headers', 'footers', 'quotes'),
            random_state=42
        )
        return dataset.data, 20
        
    elif dataset_name == 'bbc':
        url = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
        path = download_dataset(url, "bbc")
        print(f"Found dataset at: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset directory not found at {path}")
            
        dataset = load_files(path, encoding='latin-1')
        return dataset.data, 5
        
    elif dataset_name == 'reuters':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz"
        path = download_dataset(url, "reuters")
        texts = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.sgm'):
                    with open(os.path.join(root, file), 'r', encoding='latin-1') as f:
                        content = f.read()
                        start = content.find('<BODY>')
                        end = content.find('</BODY>')
                        if start != -1 and end != -1:
                            text = content[start+6:end].strip()
                            if text:
                                texts.append(text)
        
        if not texts:
            raise ValueError("No valid texts found in Reuters dataset")
            
        return texts, 8

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def evaluate_model(topics, corpus_bow, dictionary, texts):
    cm_umass = CoherenceModel(
        topics=topics,
        corpus=corpus_bow,
        dictionary=dictionary,
        coherence='u_mass'
    )

    cm_cv = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    
    return {
        'umass': cm_umass.get_coherence(),
        'c_v': cm_cv.get_coherence()
    }

def print_dataset_results(dataset_name, results):
    print(f"\nResults for {dataset_name.upper()} dataset:")
    print(f"Number of topics: {results['n_topics']}")
    
    table_data = [
        ['Custom LDA', f"{results['custom_time']:.2f}s", f"{results['custom_umass']:.4f}", f"{results['custom_cv']:.4f}"],
        ['Sklearn LDA', f"{results['sklearn_time']:.2f}s", f"{results['sklearn_umass']:.4f}", f"{results['sklearn_cv']:.4f}"]
    ]
    
    headers = ['Implementation', 'Time', 'UMass', 'C_v']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    print("\nTop words for each topic:")
    for i, topic in enumerate(results['topics']):
        print(f"Topic {i+1}: {', '.join(topic[:5])}")

def process_dataset(dataset_name):
    texts, n_topics = load_dataset(dataset_name)

    texts_tokenized = [doc.split() for doc in texts]
    dictionary = Dictionary(texts_tokenized)
    corpus_bow = [dictionary.doc2bow(t) for t in texts_tokenized]

    vectorizer = CountVectorizer(
        min_df=5,
        max_df=0.5,
        stop_words='english',
        max_features=30000
    )
    X = vectorizer.fit_transform(texts)

    print("Training custom LDA")
    start = time()
    custom_lda = LDAModel(n_topics=n_topics, n_iter=50, alpha=0.1, beta=0.01)
    custom_lda.fit(X)
    custom_time = time() - start

    custom_topics = custom_lda.get_topics(vectorizer.get_feature_names_out())
    custom_metrics = evaluate_model(custom_topics, corpus_bow, dictionary, texts_tokenized)

    print("Training sklearn LDA")
    start = time()
    sklearn_lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50, random_state=33)
    sklearn_lda.fit(X)
    sklearn_time = time() - start

    sklearn_topics = [[vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-10-1:-1]]
                     for topic in sklearn_lda.components_]
    sklearn_metrics = evaluate_model(sklearn_topics, corpus_bow, dictionary, texts_tokenized)
    
    return {
        'dataset': dataset_name,
        'n_topics': n_topics,
        'custom_time': custom_time,
        'sklearn_time': sklearn_time,
        'custom_umass': custom_metrics['umass'],
        'custom_cv': custom_metrics['c_v'],
        'sklearn_umass': sklearn_metrics['umass'],
        'sklearn_cv': sklearn_metrics['c_v'],
        'topics': custom_topics
    }

def main():
    datasets = ['20newsgroups', 'bbc', 'reuters', 'agnews']
    results = []
    
    for dataset in datasets:
        try:
            result = process_dataset(dataset)
            results.append(result)
            print_dataset_results(dataset, result)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue

    print("\nSummary Comparison:")
    summary_data = []
    for r in results:
        summary_data.append([
            r['dataset'],
            r['n_topics'],
            f"{r['custom_time']:.2f}s",
            f"{r['sklearn_time']:.2f}s",
            f"{r['custom_umass']:.4f}",
            f"{r['sklearn_umass']:.4f}",
            f"{r['custom_cv']:.4f}",
            f"{r['sklearn_cv']:.4f}"
        ])
    
    headers = [
        'Dataset',
        'Topics',
        'Custom Time',
        'Sklearn Time',
        'Custom UMass',
        'Sklearn UMass',
        'Custom C_v',
        'Sklearn C_v'
    ]
    
    print(tabulate(summary_data, headers=headers, tablefmt='grid'))

if __name__ == "__main__":
    main()
    
