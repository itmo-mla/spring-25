# Lab Work №1

## The used dataset:

In this lab The [Laptop Price Dataset](https://www.kaggle.com/datasets/ironwolf437/laptop-price-dataset) was used. This is a dataset that contains data aimed at predicting the price of laptops based on their specifications. The feature **Price (Euro)** was the target feature.

The dataset was preprocessed in the following steps:

1. Using `LabelEncoder` to encode the features with `object` values.
2. Using the `MinMaxScaler` to scale the feature values.
3. Split the dataset into **_Features_** and **_Target_**.
4. Split the data into **_Training Data_** (80%) and **_Testing Data_** (20%).

## Reference Bagging Regressor

For implementing Bagging Regressor with a library we used `BaggingRegressor` from the library `sklearn.ensemble`.

### The Results:

<img src="assets\br_ref_results.png">

### The Metrics:

_Execution Time_: 139385 mcs

_Mean R2 Score (Cross-Validation with `10` folds)_: 0.8433

## Reference Bagging Regressor

For the Custom Bagging Regressor with we implemented the class `CustomBaggingRegressor`.

### The Results:

<img src="assets\br_cus_results.png">

### The Metrics:

_Execution Time_: 541661 mcs

_Mean R2 Score (Cross-Validation with `10` folds)_: 0.8421

## Conclusions

1. We found that the **_number of estimators = 20_** for the reference implementation had the **_Mean R2 Score = 0.8433_**.
2. We found that the **_number of estimators = 20_** for the custom implementation had the **_Mean R2 Score = 0.8421_**.
3. The reference implementation was almost the same as the custom implementation according to the **_Mean R2 Score_**
4. The reference implementation was faster than the custom implementation according to the **_Execution Time_**
