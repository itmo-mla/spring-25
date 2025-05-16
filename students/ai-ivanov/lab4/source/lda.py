import structlog
import numpy as np


logger = structlog.get_logger(__name__)


class LDA:
    def __init__(
        self,
        n_topics: int,
        n_iter: int = 2000,
        alpha: float = 0.1,
        beta: float = 0.01,
        random_state: int | None = None,
    ):
        self.n_topics: int = n_topics
        self.n_iter: int = n_iter
        self.alpha: float = alpha
        self.beta: float = beta
        self.random_state: int | None = random_state
        self._rng = np.random.default_rng(random_state)

        self.vocab_: list[str] = []
        self.word2id_: dict[str, int] = {}
        self.n_vocab_: int = 0
        self.n_documents_: int = 0

        self.doc_topic_counts_: np.ndarray | None = None
        self.topic_word_counts_: np.ndarray | None = None
        self.doc_lengths_: np.ndarray | None = None
        self.topic_counts_: np.ndarray | None = None

        self.topic_word_distribution_: np.ndarray | None = None
        self.doc_topic_distribution_: np.ndarray | None = None

    def _build_vocab(self, documents: list[list[str]]) -> list[list[int]]:
        all_words = [word for doc in documents for word in doc]
        self.vocab_ = sorted(list(set(all_words)))
        self.word2id_ = {word: i for i, word in enumerate(self.vocab_)}
        self.n_vocab_ = len(self.vocab_)

        docs_as_ids = []
        for doc in documents:
            docs_as_ids.append(
                [self.word2id_[word] for word in doc if word in self.word2id_]
            )
        return docs_as_ids

    def _initialize_counts(self, documents_as_ids: list[list[int]]):
        self.n_documents_ = len(documents_as_ids)

        self.doc_topic_counts_ = np.zeros(
            (self.n_documents_, self.n_topics), dtype=np.int32
        )
        self.topic_word_counts_ = np.zeros(
            (self.n_topics, self.n_vocab_), dtype=np.int32
        )
        self.doc_lengths_ = np.array(
            [len(doc) for doc in documents_as_ids], dtype=np.int32
        )
        self.topic_counts_ = np.zeros(self.n_topics, dtype=np.int32)

        self._topic_assignments: list[list[int]] = []

        for i, doc in enumerate(documents_as_ids):
            doc_topic_assignments = []
            for word_id in doc:
                topic = self._rng.integers(0, self.n_topics)
                doc_topic_assignments.append(topic)

                self.doc_topic_counts_[i, topic] += 1
                self.topic_word_counts_[topic, word_id] += 1
                self.topic_counts_[topic] += 1
            self._topic_assignments.append(doc_topic_assignments)

    def fit(self, documents: list[list[str]]):
        """
        Learn the LDA model from the given documents.

        Parameters
        ----------
        documents : list[list[str]]
            A list of documents, where each document is a list of words (strings).
        """
        docs_as_ids = self._build_vocab(documents)
        self._initialize_counts(docs_as_ids)

        if (
            self.doc_topic_counts_ is None
            or self.topic_word_counts_ is None
            or self.doc_lengths_ is None
            or self.topic_counts_ is None
        ):
            raise RuntimeError("Counts were not initialized properly.")

        for iteration in range(self.n_iter):
            if iteration % 5 == 0:
                logger.info("Iteration", iteration=iteration, n_iter=self.n_iter)

            for d_idx, doc in enumerate(docs_as_ids):
                for w_idx, word_id in enumerate(doc):
                    current_topic = self._topic_assignments[d_idx][w_idx]

                    self.doc_topic_counts_[d_idx, current_topic] -= 1
                    self.topic_word_counts_[current_topic, word_id] -= 1
                    self.topic_counts_[current_topic] -= 1

                    numerator_doc_topic = self.doc_topic_counts_[d_idx, :] + self.alpha

                    denominator_doc_topic = (self.doc_lengths_[d_idx] - 1) + (
                        self.n_topics * self.alpha
                    )

                    if denominator_doc_topic == 0:
                        denominator_doc_topic = 1e-9

                    doc_topic_prob = numerator_doc_topic / denominator_doc_topic

                    numerator_topic_word = (
                        self.topic_word_counts_[:, word_id] + self.beta
                    )

                    denominator_topic_word = self.topic_counts_[:] + (
                        self.n_vocab_ * self.beta
                    )
                    denominator_topic_word[denominator_topic_word == 0] = 1e-9

                    topic_word_prob = numerator_topic_word / denominator_topic_word

                    conditional_prob = doc_topic_prob * topic_word_prob

                    sum_conditional_prob = np.sum(conditional_prob)
                    if sum_conditional_prob == 0 or not np.isfinite(
                        sum_conditional_prob
                    ):
                        logger.warning(
                            "Sum of conditional probabilities is 0 or not finite",
                            sum_conditional_prob=sum_conditional_prob,
                        )
                        conditional_prob = np.ones(self.n_topics) / self.n_topics
                    else:
                        conditional_prob /= sum_conditional_prob

                    new_topic = self._rng.choice(self.n_topics, p=conditional_prob)

                    self._topic_assignments[d_idx][w_idx] = new_topic

                    self.doc_topic_counts_[d_idx, new_topic] += 1
                    self.topic_word_counts_[new_topic, word_id] += 1
                    self.topic_counts_[new_topic] += 1

        denom_topic_word_dist = self.topic_counts_[:, np.newaxis] + (
            self.n_vocab_ * self.beta
        )
        denom_topic_word_dist[denom_topic_word_dist == 0] = 1e-9
        self.topic_word_distribution_ = (
            self.topic_word_counts_ + self.beta
        ) / denom_topic_word_dist

        denom_doc_topic_dist = self.doc_lengths_[:, np.newaxis] + (
            self.n_topics * self.alpha
        )
        denom_doc_topic_dist[self.doc_lengths_ == 0, :] = self.n_topics * self.alpha
        denom_doc_topic_dist[denom_doc_topic_dist == 0] = 1e-9

        self.doc_topic_distribution_ = (
            self.doc_topic_counts_ + self.alpha
        ) / denom_doc_topic_dist

        return self

    def get_topics(self, top_n_words: int = 10) -> list[list[tuple[str, float]]]:
        """
        Returns the top N words for each topic.

        Parameters
        ----------
        top_n_words : int, default=10
            The number of most probable words to return for each topic.

        Returns
        -------
        list[list[tuple[str, float]]]
            A list of topics, where each topic is a list of (word, probability) tuples.
        """
        if self.topic_word_distribution_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        topics = []
        for topic_idx in range(self.n_topics):
            word_probs = self.topic_word_distribution_[topic_idx, :]
            top_word_indices = np.argsort(word_probs)[::-1][:top_n_words]
            topic_words = [
                (self.vocab_[word_idx], word_probs[word_idx])
                for word_idx in top_word_indices
            ]
            topics.append(topic_words)
        return topics

    def transform(
        self,
        documents: list[list[str]],
        n_iterations_override: int | None = None,  # New parameter
    ) -> np.ndarray | None:
        if (
            self.topic_word_distribution_ is None
            or self.vocab_ is None
            or self.word2id_ is None
        ):
            logger.error(
                "Model has not been fitted or vocabulary is empty. Call fit() first."
            )
            return None

        new_docs_as_ids: list[list[int]] = []
        for doc in documents:
            new_docs_as_ids.append(
                [self.word2id_[word] for word in doc if word in self.word2id_]
            )

        n_new_documents = len(new_docs_as_ids)
        if n_new_documents == 0:
            return np.array([]).reshape(0, self.n_topics)

        doc_topic_counts_new = np.zeros(
            (n_new_documents, self.n_topics), dtype=np.int32
        )
        doc_lengths_effective_new = np.array(
            [len(doc_ids) for doc_ids in new_docs_as_ids], dtype=np.int32
        )

        topic_assignments_new: list[list[int]] = []

        for i, doc_ids in enumerate(new_docs_as_ids):
            doc_topic_assignments = []
            if not doc_ids:
                topic_assignments_new.append([])
                continue
            for _ in doc_ids:
                topic = self._rng.integers(0, self.n_topics)
                doc_topic_assignments.append(topic)
                doc_topic_counts_new[i, topic] += 1
            topic_assignments_new.append(doc_topic_assignments)

        if n_iterations_override is not None:
            n_transform_iter = max(1, n_iterations_override)
        else:
            n_transform_iter = max(1, self.n_iter // 20 if self.n_iter > 0 else 20)

        for iteration in range(n_transform_iter):
            if iteration % 100 == 0:
                logger.info(
                    "Transform iteration",
                    iteration=iteration,
                    n_transform_iter=n_transform_iter,
                )

            for d_idx, doc_ids in enumerate(new_docs_as_ids):
                if not doc_ids:
                    continue
                for w_idx, word_id in enumerate(doc_ids):
                    current_topic = topic_assignments_new[d_idx][w_idx]

                    doc_topic_counts_new[d_idx, current_topic] -= 1

                    numerator_doc_topic = doc_topic_counts_new[d_idx, :] + self.alpha

                    current_doc_effective_length = doc_lengths_effective_new[d_idx] - 1
                    if current_doc_effective_length < 0:
                        current_doc_effective_length = 0

                    denominator_doc_topic = current_doc_effective_length + (
                        self.n_topics * self.alpha
                    )
                    if denominator_doc_topic == 0:
                        denominator_doc_topic = 1e-9

                    doc_topic_prob = numerator_doc_topic / denominator_doc_topic

                    topic_word_prob = self.topic_word_distribution_[:, word_id]

                    conditional_prob = doc_topic_prob * topic_word_prob

                    sum_conditional_prob = np.sum(conditional_prob)
                    if sum_conditional_prob == 0 or not np.isfinite(
                        sum_conditional_prob
                    ):
                        conditional_prob = np.ones(self.n_topics) / self.n_topics
                    else:
                        conditional_prob /= sum_conditional_prob

                    new_topic = self._rng.choice(self.n_topics, p=conditional_prob)

                    topic_assignments_new[d_idx][w_idx] = new_topic
                    doc_topic_counts_new[d_idx, new_topic] += 1

        denom_doc_topic_dist_new = doc_lengths_effective_new[:, np.newaxis] + (
            self.n_topics * self.alpha
        )
        denom_doc_topic_dist_new[doc_lengths_effective_new == 0, :] = (
            self.n_topics * self.alpha
        )
        denom_doc_topic_dist_new[denom_doc_topic_dist_new == 0] = 1e-9  # Fallback

        doc_topic_distribution_new = (
            doc_topic_counts_new + self.alpha
        ) / denom_doc_topic_dist_new

        return doc_topic_distribution_new

    def perplexity(
        self, documents: list[list[str]], n_transform_iter_override: int | None = None
    ) -> float | None:
        """
        Calculates the perplexity of the model on a given set of documents.
        Perplexity is defined as exp(-log-likelihood per word).
        A lower perplexity score indicates a better model.

        Parameters
        ----------
        documents : list[list[str]]
            A list of documents (test set), where each document is a list of words.
        n_transform_iter_override : int | None, default=None
            Optional override for the number of iterations used in the transform step
            for inferring topic distributions on the test documents. If None,
            the default from the transform method is used.

        Returns
        -------
        float | None
            The perplexity score. Returns None if the model has not been fitted
            or if the test set is empty or contains no known words leading to
            a total_word_count of 0.
        """
        if (
            self.topic_word_distribution_ is None
            or self.vocab_ is None
            or self.word2id_ is None
        ):
            logger.error(
                "Model has not been fitted. Call fit() first to build topic-word distributions and vocabulary."
            )
            return None

        docs_as_ids_for_perplexity: list[list[int]] = []
        for doc_text in documents:
            doc_ids = [
                self.word2id_[word] for word in doc_text if word in self.word2id_
            ]
            docs_as_ids_for_perplexity.append(doc_ids)

        if not documents:
            logger.info("Input documents list for perplexity is empty.")
            return None

        if not any(docs_as_ids_for_perplexity):
            logger.warning(
                "All test documents became empty after vocabulary filtering. Cannot calculate perplexity."
            )
            return None

        doc_topic_dist_test = self.transform(
            documents, n_iterations_override=n_transform_iter_override
        )

        if doc_topic_dist_test is None or doc_topic_dist_test.shape[0] == 0:
            logger.warning(
                "Transforming test documents for perplexity calculation failed or yielded empty result."
            )
            return None

        if doc_topic_dist_test.shape[0] != len(docs_as_ids_for_perplexity):
            logger.error(
                "Mismatch in document counts between processed IDs and transform output during perplexity calculation."
            )
            return None

        log_likelihood_sum = 0.0
        total_word_count = 0

        for doc_idx, current_doc_ids in enumerate(docs_as_ids_for_perplexity):
            if not current_doc_ids:
                continue

            current_doc_topic_dist = doc_topic_dist_test[doc_idx, :]

            for word_id in current_doc_ids:
                prob_word_given_topic = self.topic_word_distribution_[:, word_id]

                prob_word_given_doc = np.sum(
                    prob_word_given_topic * current_doc_topic_dist
                )

                if prob_word_given_doc > 1e-12:
                    log_likelihood_sum += np.log(prob_word_given_doc)
                else:
                    log_likelihood_sum += np.log(1e-12)
                    logger.debug(
                        f"Word ID {word_id} in doc {doc_idx} (perplexity calc) had near-zero probability ({prob_word_given_doc})."
                    )
                total_word_count += 1

        if total_word_count == 0:
            logger.warning(
                "No words from the test documents were processed (e.g., all filtered out or test set effectively empty). "
                "Total word count is 0. Cannot calculate perplexity."
            )
            return None

        mean_log_likelihood = log_likelihood_sum / total_word_count

        if not np.isfinite(mean_log_likelihood):
            logger.warning(
                f"Mean log likelihood is not finite: {mean_log_likelihood}. Perplexity may be inf or NaN."
            )

        perplexity_score = np.exp(-mean_log_likelihood)

        if not np.isfinite(perplexity_score):
            logger.warning(
                f"Calculated perplexity is not finite: {perplexity_score}. Mean LL: {mean_log_likelihood}"
            )

        return perplexity_score


if __name__ == "__main__":
    documents_data = [
        ["apple", "banana", "fruit", "sweet", "healthy"],
        ["computer", "science", "programming", "code", "software"],
        ["banana", "fruit", "yellow", "sweet", "healthy"],
        ["apple", "computer", "fruit", "code", "algorithm"],
        ["programming", "software", "developer", "code", "science"],
        ["healthy", "food", "fruit", "vegetable", "diet"],
        ["sweet", "apple", "banana", "food", "recipe"],
        ["science", "research", "computer", "algorithm", "data"],
    ]

    print("Initializing LDA model...")
    lda_model = LDA(n_topics=3, n_iter=100, random_state=42, alpha=0.1, beta=0.1)

    print("Fitting LDA model...")
    lda_model.fit(documents_data)

    print("\nLearned Topics (Top 5 words):")
    topics = lda_model.get_topics(top_n_words=5)
    for i, topic in enumerate(topics):
        print(f"Topic {i + 1}: {[(word, f'{prob:.4f}') for word, prob in topic]}")

    print("\nDocument-Topic Distributions (first 3 docs):")
    if lda_model.doc_topic_distribution_ is not None:
        for i in range(min(3, len(documents_data))):
            print(f"Document {i + 1}: {lda_model.doc_topic_distribution_[i, :]}")

    print("\nTransforming new documents:")
    new_docs = [
        ["sweet", "fruit", "recipe", "healthy"],
        ["code", "algorithm", "software", "system"],
    ]
    transformed_dist = lda_model.transform(new_docs)
    if transformed_dist is not None:
        for i, doc_dist in enumerate(transformed_dist):
            print(f"New Document {i + 1} topic distribution: {doc_dist}")

    print("\nTesting with a document containing only unknown words:")
    unknown_doc = [["unknown_word_1", "unknown_word_2"]]
    transformed_unknown = lda_model.transform(unknown_doc)
    if transformed_unknown is not None:
        print(f"Unknown Document topic distribution: {transformed_unknown}")

    print("\nTesting with an empty document list for transform:")
    transformed_empty_list = lda_model.transform([])
    if transformed_empty_list is not None:
        print(f"Transform empty list result: {transformed_empty_list}")

    print("\nTesting with a document list containing an empty document for transform:")
    transformed_with_empty_doc = lda_model.transform(
        [["word1"], [], ["word2", "word1"]]
    )
    if transformed_with_empty_doc is not None:
        print(f"Transform with empty doc result: {transformed_with_empty_doc}")

    print("\nCalculating perplexity on the training data (for demonstration):")
    perplexity_train = lda_model.perplexity(documents_data)
    if perplexity_train is not None:
        print(f"Perplexity on training data: {perplexity_train:.4f}")

    print("\nCalculating perplexity on new documents (test data):")
    test_docs_for_perplexity = [
        ["sweet", "fruit", "recipe", "healthy", "banana"],
        ["code", "algorithm", "software", "system", "computer", "science"],
        ["food", "diet", "apple", "vegetable"],
        ["unknown", "words", "only"],  # This doc will have 0 known words
        [],  # Empty document
    ]
    perplexity_test = lda_model.perplexity(test_docs_for_perplexity)
    if perplexity_test is not None:
        print(f"Perplexity on new test data: {perplexity_test:.4f}")

    print("\nCalculating perplexity on an empty list of documents:")
    perplexity_empty_set = lda_model.perplexity([])
    if perplexity_empty_set is None:
        print("Perplexity on empty set is None (as expected).")

    print("\nCalculating perplexity on documents with only unknown words:")
    perplexity_unknown_set = lda_model.perplexity(
        [["unknown", "unseen"], ["wordx", "wordy"]]
    )
    if perplexity_unknown_set is None:  # Expect None as total_word_count will be 0
        print("Perplexity on set with only OOV words is None (as expected).")
