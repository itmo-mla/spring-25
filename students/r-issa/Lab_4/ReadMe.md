# Latent Dirichlet Allocation (LDA) - Lab Report

## Objective

The goal of this lab was to implement the **Latent Dirichlet Allocation (LDA)** algorithm from scratch and compare it with a reference implementation from the **scikit-learn** library. We evaluated both models using **topic coherence** and **training time**, using a real-world dataset.

---

## Dataset

We used the [Movie Genre Classification dataset](https://www.kaggle.com/datasets/therohithanand/movie-genre-classification) from Kaggle. The dataset contains 50,000 movie descriptions with metadata, including:

- `Description`: short movie plot summary (used for topic modeling)
- `Genre`: target label (not used directly in LDA since it's unsupervised)

Preprocessing steps included:

- Lowercasing
- Removing digits and punctuation
- Removing stopwords
- Lemmatization using spaCy
- Filtering tokens with length <= 2

We also used TF-IDF and CountVectorizer representations.

---

## Reference Implementation: Scikit-learn LDA

### Configuration

- Library: `sklearn.decomposition.LatentDirichletAllocation`
- Topics: 7
- Vectorizer: TF-IDF
- Iterations: 10
- Learning method: `online`

### Results

- **Execution Time**: 97,765 microseconds
- **Topic Coherence (c_v)**: **0.2307**

Example topics:

```
Topic #1:
love touching moment heartwarming story fill guarantee laughter comedy hearted

Topic #2:
journey character explore complex emotional fill light laughter hearted comedy

Topic #3:
action thriller intense pace scene fast fill guarantee light laughter

Topic #4:
evoke fear tale spine chilling dread fill light comedy laughter

Topic #5:
unexpected twist suspenseful plot light hearted comedy laughter guarantee fill

Topic #6:
fill suspenseful twist unexpected plot guarantee hearted laughter comedy light

Topic #7:
world wonder magic imaginative fill twist unexpected plot suspenseful light
```

---

## Custom Implementation: Gibbs Sampling LDA

We implemented a simplified LDA model using Gibbs Sampling. The class `CustomLatentDirichletAllocation` uses document-topic and topic-word count matrices and reassigns topic labels over multiple iterations.

### Configuration

- Topics: 7
- Iterations: 100
- Alpha: 0.1
- Beta: 0.01

### Results

- **Execution Time**: 847,394 microseconds
- **Topic Coherence (c_v)**: **0.2269**

Example topics:

```
Topic #1: laughter light comedy guarantee hearted fill magic wonder world imaginative

Topic #2: fast intense scene thriller pace action fill touching story spine

Topic #3: tale spine fear evoke dread chilling love moment story touching

Topic #4: journey explore emotional complex character heartwarming love moment story touching

Topic #5: fill unexpected twist suspenseful plot intense emotional complex character thriller

Topic #6: action pace thriller scene intense fast fill touching story spine

Topic #7: imaginative world fill magic wonder spine tale chilling dread evoke
```

---

## Comparison Summary

| Model         | Method            | Iterations | Coherence (c_v) | Execution Time (μs) |
| ------------- | ----------------- | ---------- | --------------- | ------------------- |
| Reference LDA | Variational Bayes | 10         | **0.2307**      | 97,765              |
| Custom LDA    | Gibbs Sampling    | 100        | 0.2269          | 847,394             |

---

## Conclusion

- The **reference LDA** from scikit-learn achieved **higher topic coherence**, likely due to TF-IDF input and fast variational inference.
- The **custom LDA**, while functional, showed lower coherence due to limited sampling iterations, random initialization, and lack of burn-in.

### Future Improvements

- Increase Gibbs sampling iterations further (e.g., 500)
- Add burn-in and averaging
- Tune hyperparameters `alpha`, `beta`
- Visualize topic distributions (e.g., with pyLDAvis)
