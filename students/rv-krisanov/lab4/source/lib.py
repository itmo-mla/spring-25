import numpy as np


def customLDA(
    doc_term_matrix: np.ndarray,  # [document_index, word_index] = word_count
    topic_count: int = 5,
    alpha: float = 1,
    beta: float = 0.1,
    iteration: int = 50,
):
    doc_term_matrix = doc_term_matrix.astype(np.int32)
    [D, V], T = (
        doc_term_matrix.shape,
        topic_count,
    )  # documents, vocabulary size, topic count

    phi = np.zeros((T, V), dtype=np.int32)  # темы × слова
    theta = np.zeros((D, T), dtype=np.int32)  # документы × темы

    rows, cols = np.where(doc_term_matrix > 0)
    counts = doc_term_matrix[rows, cols]

    token_document_map = np.repeat(rows, counts)  # (N_tokens,)
    token_word_map = np.repeat(cols, counts)
    token_topic_map = np.random.randint(
        0, topic_count, token_document_map.size, dtype=np.int32
    )

    token_idx_ptr = np.zeros(D + 1, dtype=np.int32)
    token_idx_ptr[1:] = np.cumsum(doc_term_matrix.sum(axis=1))
    word_per_topics = np.zeros(
        shape=(V, T), dtype=np.int32
    )  # отдельно каждое слово в каждом топике
    np.add.at(word_per_topics, (token_word_map, token_topic_map), 1)
    words_count_by_topic = np.bincount(
        token_topic_map, minlength=T
    )  # сколько слов в топике
    word_topic_by_document = np.zeros(
        shape=(D, T), dtype=np.int32
    )  # сколько всего слов в документе помечено темой
    np.add.at(word_topic_by_document, (token_document_map, token_topic_map), 1)
    for _iter in range(1, iteration):
        for token_idx in np.random.permutation(token_document_map.shape[0]):
            document = token_document_map[token_idx]
            word = token_word_map[token_idx]
            topic = token_topic_map[token_idx]

            # Forgetting...
            word_per_topics[word, topic] -= 1
            words_count_by_topic[topic] -= 1
            word_topic_by_document[document, topic] -= 1
            # new topic choosing
            new_topic_probability = (
                (word_per_topics[word].astype(np.float32) + beta)
                / (words_count_by_topic.astype(np.float32) + V * beta)
                * (word_topic_by_document[document].astype(np.float32) + alpha)
            )
            new_topic_probability /= new_topic_probability.sum()
            new_topic = np.random.choice(T, p=new_topic_probability)
            # membering new topic
            word_per_topics[word, new_topic] += 1
            words_count_by_topic[new_topic] += 1
            word_topic_by_document[document, new_topic] += 1

            token_topic_map[token_idx] = new_topic
    np.add.at(
        phi,
        (
            token_topic_map,
            token_word_map,
        ),
        1 / token_word_map.shape[0],
    )
    np.add.at(
        theta,
        (
            token_document_map,
            token_topic_map,
        ),
        1 / token_word_map.shape[0],
    )

    phi = phi.astype(np.float32)
    theta = theta.astype(np.float32)

    topic_sizes = words_count_by_topic.astype(np.float32)  # n_{z,·}
    phi = (word_per_topics.T + beta) / (topic_sizes[:, None] + V * beta)  # T×V

    doc_lens = word_topic_by_document.sum(axis=1, keepdims=True)
    theta = (word_topic_by_document + alpha) / (doc_lens + T * alpha)  # D×T
    return phi, theta
