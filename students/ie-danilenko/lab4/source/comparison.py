from read import read_texts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from time import time
from lda import LDA
from metrics import perplexity

texts = read_texts('data/cleaned_texts.csv')
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(texts)

start = time()
sk_lda = LatentDirichletAllocation(n_components=3, random_state=42, learning_method='online')
sk_lda.fit(X)
vocab = vectorizer.get_feature_names_out()
print("Темы (sklearn LDA):")
for idx, topic in enumerate(sk_lda.components_):
    top_words = [vocab[i] for i in topic.argsort()[-10:]]
    print(f"Тема {idx+1}: {', '.join(top_words)}")

print(f"Sklearn time: {time() - start}")
print(f"Sklearn metrics {perplexity(sk_lda, X)}")

print()
start = time()
lda = LDA(n_components=3, max_iter=50, random_state=42)
lda.fit(X)

topics = lda.get_topics(vectorizer, 10)
for i, t in enumerate(topics):
    print(f"Topic {i+1}:", ", ".join(t))

print(f"My LDA time: {time() - start}")
print(f"My LDA metrics {perplexity(lda, X)}")

