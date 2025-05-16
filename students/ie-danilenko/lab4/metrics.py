import numpy as np

def perplexity(model, X):
    if hasattr(model, 'components_'):
        log_proba = model.components_
        doc_logprob = X @ log_proba.T
        log_likelihood = doc_logprob.sum()
        perplexity = np.exp(-log_likelihood / X.sum())
        return perplexity
    else:
        raise ValueError("Model doesn't have components_ attribute.")