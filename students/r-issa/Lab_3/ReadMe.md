# Lab 3: Naive Bayes Classifier

## 📌 Objective

This lab aimed to implement a custom **Multinomial Naive Bayes classifier** and compare its performance with the reference implementation from **scikit-learn**. The classifier was applied to a real-world text classification problem — detecting spam SMS messages.

---

## 🧠 Algorithm Description

**Naive Bayes Classifier** is a probabilistic machine learning model based on Bayes’ Theorem:

$$
P(C_k | x) = \frac{P(x | C_k) \cdot P(C_k)}{P(x)}
$$

Assuming independence between features, this becomes efficient for high-dimensional data like text.

We used the **Multinomial Naive Bayes** variant, appropriate for discrete count features (e.g., word frequencies in documents). The model calculates:

- **Prior**: $( P(C_k) )$
- **Likelihood**: $( P(w_i | C_k) )$ with Laplace smoothing:

  $$
  P(w_i | C_k) = \frac{\text{count}(w_i, C_k) + \alpha}{\sum_j \text{count}(w_j, C_k) + \alpha \cdot V}
  $$

---

## 📂 Dataset Description

- **Name**: SMS Spam Collection Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size**: 5,572 messages
- **Classes**:
  - `ham` (not spam): 0
  - `spam`: 1

**Preprocessing Steps**:

- Converted labels to binary (ham → 0, spam → 1)
- Cleaned text (lowercased, removed punctuation)
- Transformed messages to Bag-of-Words vectors using `CountVectorizer`

---

## 🛠️ Implementation

### ✅ Reference Classifier

- **Library**: `sklearn.naive_bayes.MultinomialNB`
- **Accuracy**: `0.9785`
- **10-fold Cross-Validation Accuracy**: `0.9772`
- **Training Time**: `3,311 µs`

### ✅ Custom Classifier

- Implemented from scratch
- Handles:
  - Prior probabilities
  - Conditional probabilities with Laplace smoothing
  - Predicts using log posterior
- **Accuracy**: `0.9785`
- **10-fold Cross-Validation Accuracy**: `0.9772`
- **Training Time**: `373,898 µs`

---

## 📈 Results

| Metric              | Reference (`sklearn`) | Custom Implementation |
| ------------------- | --------------------- | --------------------- |
| Accuracy            | 0.9785                | 0.9785                |
| 10-fold CV Accuracy | 0.9772                | 0.9772                |
| Training Time (µs)  | 3,311                 | 373,898               |

---

## ✅ Conclusion

- Both classifiers achieved nearly identical accuracy and classification metrics.
- The custom implementation took significantly longer to train due to manual array handling and lack of optimization.
- The experiment validates the correctness of the custom Naive Bayes logic and demonstrates the efficiency of using optimized libraries for large-scale tasks.
