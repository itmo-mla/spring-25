# Lab Work №1

<style>
    .highlighted_table {
        text-align: center;
    
    }
    .highlighted_table.element_1 tr:nth-child(1) { 
        color: black;
        background: rgb(150, 150, 150); 
        }
    .highlighted_table.element_2 tr:nth-child(2) { 
        color: black;
        background: rgb(150, 150, 150); 
        }
    .highlighted_table.element_3 tr:nth-child(3) { 
        color: black;
        background: rgb(150, 150, 150); 
        }
    .highlighted_table.element_4 tr:nth-child(4) { 
        color: black;
        background: rgb(150, 150, 150); 
        }
    .highlighted_table.element_5 tr:nth-child(5) { 
        color: black;
        background: rgb(150, 150, 150); 
        }
</style>

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

_The best result is highlighted in the table_

<div class="highlighted_table element_5">

| Estimators | Execution Time | R2 Score |
| ---------- | -------------- | -------- |
| 10         | 3579 mcs       | 0.8651   |
| 20         | 8200 mcs       | 0.8656   |
| 30         | 9464 mcs       | 0.8660   |
| 40         | 12447 mcs      | 0.8653   |
| 50         | 13173 mcs      | 0.8716   |

</div>

## Reference Bagging Regressor

For the Custom Bagging Regressor with we implemented the class `CustomBaggingRegressor`.

### The Results:

<img src="assets\br_cus_results.png">

### The Metrics:

_The best result is highlighted in the table_

<div class="highlighted_table element_4">

| Estimators | Execution Time | R2 Score |
| ---------- | -------------- | -------- |
| 10         | 2427 mcs       | 0.8584   |
| 20         | 4646 mcs       | 0.8569   |
| 30         | 7461 mcs       | 0.8504   |
| 40         | 9890 mcs       | 0.8648   |
| 50         | 11744 mcs      | 0.8594   |

</div>

## Conclusions

1. We found that the best **_number of estimators_** for the reference implementation was `50` with the **_R2 Score = 0.8716_**.
2. We found that the best **_number of estimators_** for the custom implementation was `40` with the **_R2 Score = 0.8648_**.
3. the reference implementation was better than the custom implementation according to the **_R2 Score_**
4. the custom implementation was faster than the reference implementation according to the **_Execution Time_**
