import numpy as np
from sklearn.decomposition import LatentDirichletAllocation # Импортируем для проверки типа модели

def perplexity(model, X):
    n_samples, n_features = X.shape
    log_likelihood = 0
    total_words = 0

    # Проверка типа модели и получение распределений
    if isinstance(model, LatentDirichletAllocation):
        # Для модели scikit-learn
        if not hasattr(model, 'components_'):
             raise ValueError("Sklearn model must have 'components_' attribute.")
        topic_word_dist = model.components_ / model.components_.sum(axis=1, keepdims=True)
        doc_topic_dist = model.transform(X)

    elif hasattr(model, 'doc_topic_counts_') and hasattr(model, 'components_'):
        # Для вашей модели LDA
        topic_word_dist = (model.topic_word_counts_ + model.beta) / (model.topic_word_counts_.sum(axis=1, keepdims=True) + model.vocab_size * model.beta)
        doc_topic_dist = (model.doc_topic_counts_ + model.alpha) / (model.doc_topic_counts_.sum(axis=1, keepdims=True) + model.n_components * model.alpha)

    else:
        raise ValueError("Unsupported model type. Model must be either sklearn LatentDirichletAllocation or have 'doc_topic_counts_' and 'components_' attributes.")

    # Вычисление логарифмической вероятности корпуса (одинаково для обеих моделей после получения распределений)
    for i in range(n_samples):
        row = X[i].toarray()[0] if hasattr(X, "toarray") else X[i]
        for j in range(n_features):
            count = int(row[j])
            if count > 0:
                total_words += count
                # Вероятность слова j в документе i по всем темам
                # Убедитесь, что оба распределения имеют правильную форму для умножения
                p_word_doc = np.sum(doc_topic_dist[i, :] * topic_word_dist[:, j])
                if p_word_doc > 0:
                    log_likelihood += count * np.log(p_word_doc)

    # Расчет perplexity
    if total_words == 0:
        return float('inf') # Избежать деления на ноль
    perplexity_value = np.exp(-log_likelihood / total_words)

    return perplexity_value