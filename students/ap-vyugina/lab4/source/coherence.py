from typing import List

import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def calculate_lda_coherence(
    topic_word_distribution,
    feature_names: List[str],
    texts: List[List[str]],
    n_words: int = 10,
    n_topics: int = 10
) -> float:
    """
    topic_word_distribution: esulting distribution from the model
    lda_model: Trained sklearn LatentDirichletAllocation model
    feature_names: List of feature names from CountVectorizer
    texts: List of tokenized documents
    n_words: Number of top words to consider for coherence calculation
    """
    dictionary = Dictionary(texts)
    
    # Get top words for each topic
    topic_words = []
    for topic in range(n_topics):
        # Get indices of top words for this topic
        top_indices = np.argsort(topic_word_distribution[topic])[::-1][:n_words]
        # Get words
        topic_words.append([feature_names[idx] for idx in top_indices])
    
    # Create coherence model
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    
    return coherence_model.get_coherence()
