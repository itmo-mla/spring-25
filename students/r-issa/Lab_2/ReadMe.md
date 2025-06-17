# Custom Gradient Boosting Regressor

## 1. Algorithm Description

This project implements a custom Gradient Boosting Regressor from scratch using NumPy and scikit-learn's `DecisionTreeRegressor` as the base learner. Gradient Boosting is an ensemble method that builds additive models in a forward stage-wise fashion. It allows optimization of an arbitrary differentiable loss function. The algorithm:

- Starts with an initial prediction (mean of the target variable).
- Iteratively fits regression trees to the residuals (pseudo-gradients) of the previous prediction.
- Updates the prediction by adding a scaled version of the new tree's predictions.

Our implementation supports the following hyperparameters:

- `n_estimators`: Number of boosting rounds (trees).
- `learning_rate`: Shrinks the contribution of each tree.
- `max_depth`: Controls the complexity of each regression tree.

## 2. Dataset Description

We used the [California Housing dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices), which includes the following features:

- `longitude`, `latitude`
- `housing_median_age`
- `total_rooms`, `total_bedrooms`
- `population`, `households`
- `median_income`
- `ocean_proximity` (categorical)

The target variable is `median_house_value`. Preprocessing steps included:

- Imputing missing values in `total_bedrooms` using the median.
- Encoding the categorical feature `ocean_proximity` using `LabelEncoder`.
- Normalizing features using `MinMaxScaler`.
- Splitting the dataset into training and test sets (80/20 split).

## 3. Experiments

We evaluated both the reference and custom Gradient Boosting Regressor using the following metrics:

- **Execution Time** (microseconds)
- **R² Score** on the test set
- **Cross-Validation R² Score** with 10-fold `KFold` cross-validation

### Reference Gradient Boosting Regressor (scikit-learn)

<img src="assets\gbr_ref_results.png">

- Number of Estimators: 100
- Execution Time: `585378` µs
- R² Score: `0.7585`
- Mean Cross-Validation R² Score: `0.7709`

### Custom Gradient Boosting Regressor

<img src="assets\gbr_cus_results.png">

- Number of Estimators: 100
- Execution Time: `440397` µs
- R² Score: `0.7585`
- Mean Cross-Validation R² Score: `0.7708`

## 4. Comparison with scikit-learn

Both implementations achieved nearly identical predictive performance (R² score and CV R²), confirming the correctness of the custom model. Interestingly, the custom implementation was slightly faster in execution time during the fit and predict steps. This may be due to reduced internal overhead in the simplified model compared to scikit-learn’s more robust implementation.

| Metric                     | scikit-learn | Custom Implementation |
| -------------------------- | ------------ | --------------------- |
| Execution Time (µs)        | `585378`     | `440397`              |
| R² Score                   | `0.7585`     | `0.7585`              |
| Mean CV R² Score (10-fold) | `0.7709`     | `0.7708`              |

## 5. Conclusion

This project demonstrated the internal workings of the Gradient Boosting algorithm through a from-scratch implementation. We successfully replicated the performance of scikit-learn’s Gradient Boosting Regressor, confirming the theoretical and practical understanding of the method. Despite its simplicity, the custom model performed comparably in both runtime and accuracy, offering a transparent view into how gradient boosting operates under the hood.

This exercise deepened our understanding of ensemble learning, regression trees, additive modeling, and gradient-based optimization in supervised learning.
