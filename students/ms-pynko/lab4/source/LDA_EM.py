import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import CoherenceModel
from tqdm import tqdm
from time import time
from sklearn.decomposition import LatentDirichletAllocation


def get_topics_sklearn(model, feature_names, n_top_words=10):
    topics = []
    for topic_weights in model.components_:
        top_idx = topic_weights.argsort()[:-n_top_words-1:-1]
        topics.append([feature_names[i] for i in top_idx])
    return topics

def get_metrics(topics, corpus_bow, dictionary, texts):
    cm_umass = CoherenceModel(topics=topics,
                            corpus=corpus_bow,
                            dictionary=dictionary,
                            coherence='u_mass')
    cm_cv    = CoherenceModel(topics=topics,
                            texts=texts,          
                            dictionary=dictionary,
                            coherence='c_v')
    print('UMass:', cm_umass.get_coherence())
    print('c_v  :', cm_cv.get_coherence())
    return cm_umass.get_coherence(), cm_cv.get_coherence()

newsgroups = fetch_20newsgroups(remove=("headers","footers","quotes"))

class LDA_EM:
    def __init__(self, n_topics: int,
                 n_iter: int = 50,
                 alpha: float = 0.1,
                 beta:  float = 0.01,
                 random_state: int = 42):
        self.K      = n_topics
        self.n_iter = n_iter
        self.alpha  = alpha
        self.beta   = beta
        if random_state is not None:
            np.random.seed(random_state)

   
    def fit(self, X):
        X = csr_matrix(X, dtype=np.float64)         
        D, V = X.shape

        self.phi   = np.random.dirichlet(np.ones(V), self.K)    
        self.theta = np.random.dirichlet(np.ones(self.K), D)     

        for it in tqdm(range(self.n_iter)):

            n_wt = np.zeros((self.K, V))
            n_td = np.zeros((D, self.K))

            rows, cols = X.nonzero()
            data       = X.data
            for idx in range(len(data)):
                d, w, cnt = rows[idx], cols[idx], data[idx]

                p = self.phi[:, w] * self.theta[d]
                p /= p.sum()

                n_wt[:, w] += cnt * p
                n_td[d]    += cnt * p

            self.phi   = n_wt + self.beta          
            self.phi  /= self.phi.sum(axis=1, keepdims=True)

            self.theta = n_td + self.alpha          
            self.theta /= self.theta.sum(axis=1, keepdims=True)

        return self

    def predict(self, X):
        if not hasattr(self, "phi"):
            raise RuntimeError("Сначала вызовите fit()")
        return np.argmax(self.transform(X), axis=1)

    def transform(self, X):
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

if __name__ == "__main__":
    
    data = newsgroups.data
    texts = [doc.split() for doc in data]
    dictionary = Dictionary(texts)      
    corpus_bow = [dictionary.doc2bow(t) for t in texts]
    vectorizer = CountVectorizer(min_df=5, max_df=0.5, stop_words='english', max_features=30000)
    X = vectorizer.fit_transform(data)
    start = time()
    lda = LDA_EM(n_topics=20, n_iter=50, alpha=0.1, beta=0.01, random_state=42)
    lda.fit(X)
    print(f"Time: {time() - start} seconds")
    topics = lda.get_topics(vectorizer.get_feature_names_out(), n_top_words=20)
    print(topics)
    get_metrics(topics, corpus_bow, dictionary, texts)
    print("-"*100)
    print("sklearn")    
    start = time()
    lda = LatentDirichletAllocation(n_components=20, max_iter=50, random_state=42)
    lda.fit(X)
    print(f"Time: {time() - start} seconds")

    end = time()
    print(f"Time: {end - start} seconds")
    topics = get_topics_sklearn(lda, vectorizer.get_feature_names_out(), n_top_words=20)
    get_metrics(topics, corpus_bow, dictionary, texts)
    
