# Lab 5: Latent Factor Model (LFM)

## Objective

The goal of this lab is to implement a custom Latent Factor Model (LFM) for collaborative filtering and compare its performance with a reference implementation using `scikit-learn`'s `TruncatedSVD`. The models are evaluated based on prediction accuracy (RMSE, MAE) and training time.

---

## Dataset Description

We used the **MovieLens 20M** dataset, which consists of 20 million ratings applied to 27,000 movies by 138,000 users. It includes:

- `rating.csv`: Contains userId, movieId, rating (0.5 to 5.0), and timestamp.
- `movie.csv`: Contains movieId, title, and genre information.

**Dataset source**: [Kaggle - MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)

For model input, the dataset was preprocessed to:

- Drop the timestamp column.
- Encode user and movie IDs as integer indices.
- Split into training (80%) and test (20%) sets.

---

## Latent Factor Model Overview

Latent Factor Models aim to represent users and items (e.g., movies) in a shared low-dimensional vector space. Each user and each item is represented by a latent vector. The predicted rating is computed as the dot product of these vectors:

$$
\hat{r}_{ui} = P_u^\top Q_i
$$

Where:

- $( P_u $): latent vector of user $( u $)
- $( Q_i $): latent vector of item $( i $)

The model is trained using **Stochastic Gradient Descent (SGD)** to minimize the squared error:

$$
\min \sum_{(u, i)} (r_{ui} - P_u^\top Q_i)^2 + \lambda (\|P_u\|^2 + \|Q_i\|^2)
$$

---

## Implementation Details

### ✅ Reference Implementation (TruncatedSVD)

- Library: `scikit-learn`
- Algorithm: `TruncatedSVD` on sparse user-item matrix
- Predicted ratings computed from reduced user and item feature matrices.

**Training time**: `~134,427 µs`  
**Evaluation metrics**:

- RMSE: `2.6756`
- MAE: `2.4302`

---

### ✅ Custom Implementation (CustomTruncatedSVD)

- Technique: Matrix factorization with SGD
- Optimized with mini-batch vectorization (batch size: 100,000)
- Latent dimension: 20
- Regularization: 0.1
- Learning rate: 0.01
- Epochs: 10

**Training time**: `~909,477 µs`  
**Evaluation metrics**:

- RMSE: `1.2390`
- MAE: `0.9036`

---

## Results Comparison

| Model                  | RMSE   | MAE    | Training Time (µs) |
| ---------------------- | ------ | ------ | ------------------ |
| TruncatedSVD (sklearn) | 2.6756 | 2.4302 | 134,427            |
| Custom LFM (SGD)       | 1.2390 | 0.9036 | 909,477            |

---

### Observations

- The **Custom LFM** significantly outperforms the `TruncatedSVD` baseline in terms of both RMSE and MAE, achieving **over 50% lower error**.
- While the custom model requires more training time (~6.7× longer), this tradeoff is acceptable given the major gain in prediction accuracy.
- `TruncatedSVD` is a linear dimensionality reduction method and does not optimize for rating prediction directly, while the custom LFM explicitly minimizes prediction error using gradient descent.

---

## Conclusions

- The custom Latent Factor Model using SGD demonstrates **superior predictive performance** on the MovieLens 20M dataset compared to the `scikit-learn` TruncatedSVD reference.
- Although training is slower due to iterative updates and manual optimization, the custom implementation is more **suitable for recommendation tasks** because it directly models user-item interactions.
- The experiment highlights the trade-off between accuracy and computational efficiency, as well as the value of implementing specialized models tailored to a specific prediction goal.

---
