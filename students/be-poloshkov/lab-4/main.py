import re
import time
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.corpora import Dictionary
from lda import LDA
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk


def main():
    data = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        categories=['comp.graphics', 'sci.electronics', 'talk.politics.guns', 'soc.religion.christian'],
    )
    documents = data.data

    stop_words = set(stopwords.words('english'))

    processed_docs = preprocess(documents, stop_words)
    processed_docs = [doc for doc in processed_docs if len(doc) > 0]

    dictionary = defaultdict()
    dictionary.default_factory = lambda: len(dictionary)
    vocab_size = 0

    corpus = []
    for doc in processed_docs:
        bow = []
        for word in doc:
            idx = dictionary[word]
            bow.append(idx)
        corpus.append(bow)
        vocab_size = max(vocab_size, max(bow) + 1)

    index_to_word = {v: k for k, v in dictionary.items()}

    lda_custom = LDA(n_topics=10, n_iter=20)
    start_time = time.time()
    lda_custom.fit(corpus, vocab_size)
    end_time = time.time()

    training_time_custom = end_time - start_time
    print("Self-made algo time:", training_time_custom)

    topics_custom = lda_custom.get_vocabulary(index_to_word)
    for i, words in enumerate(topics_custom):
        print(f"Topic {i}: {', '.join(words)}")

    # coherence

    gensim_dictionary = Dictionary(processed_docs)

    coherence_model_custom = CoherenceModel(
        topics=topics_custom,
        texts=processed_docs,
        dictionary=gensim_dictionary,
        coherence='c_v',
    )
    coherence_custom = coherence_model_custom.get_coherence()
    print("Self-made algo coherence:", coherence_custom)

    # sklearn
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = vectorizer.fit_transform(documents)

    start_time_sk = time.time()
    lda_sklearn = LatentDirichletAllocation(n_components=10, random_state=42, max_iter=20)
    lda_sklearn.fit(tf)
    end_time_sk = time.time()

    training_time_sk = end_time_sk - start_time_sk
    print("SKLEARN time:", training_time_sk)

    topics_sklearn = get_sklearn_topics(lda_sklearn, vectorizer.get_feature_names_out(), 10)

    dictionary_sk = corpora.Dictionary([vectorizer.get_feature_names_out().tolist()])

    cm_sklearn = CoherenceModel(
        topics=[
            [dictionary_sk.token2id[w] for w in topic]
            for topic in topics_sklearn
        ],
        texts=processed_docs,
        dictionary=gensim_dictionary,
        coherence='c_v',
    )
    coherence_sklearn = cm_sklearn.get_coherence()
    print("SKLEARN coherence: ", coherence_sklearn)

    for i, words in enumerate(topics_sklearn):
        print(f"Topic {i}: {', '.join(words)}")



def preprocess(texts, stop_words):
    processed = []
    for doc in texts:
        tokens = word_tokenize(re.sub(r'\W+', ' ', doc.lower()))
        filtered = [word for word in tokens if word not in stop_words and len(word) > 2]
        processed.append(filtered)
    return processed

def get_sklearn_topics(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(top_words)
    return topics


if __name__ == '__main__':
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    main()