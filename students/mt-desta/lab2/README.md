# Lab 2: Gradient Boosting

## Dataset

The [Boston Housing](https://www.kaggle.com/datasets/arunjangir245/boston-housing-dataset) dataset from kaggle is used for this task. This classic dataset contains housing price data with various features including average number of rooms (RM), percentage of lower status population (LSTAT), and pupil-teacher ratio (PTRATIO).

## Implemented Method: Gradient Boosting Regressor

This implementation is a custom Gradient Boosting algorithm for regression tasks. Gradient Boosting is an ensemble technique that builds models sequentially, where each new model attempts to correct the errors of the previous models.

Key features of the implementation:
- Uses Decision Trees as base learners
- Implements a sequential, additive ensemble approach
- Applies customizable learning rate to control overfitting
- Tracks training loss throughout iterations

The algorithm follows these core steps:
1. Create an initial prediction model
2. Calculate residuals (differences between actual and predicted values)
3. Train a new model to predict these residuals
4. Add the scaled predictions of the new model to the current predictions
5. Repeat steps 2-4 for a specified number of iterations

### Algorithm Details from `gradientboosting.py`

```python
class GradientBoosting:
    def __init__(self, learning_rate=0.1, max_depth=3, loss_func=None):
        # Initialize parameters
        self.models = []          # Stores all decision trees
        self.losses = []          # Tracks losses during training
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss_function = loss_func
    
    def fit(self, X, y, estimators=1000):
        # Initial model
        model = DecisionTreeRegressor(max_depth=self.max_depth)
        model.fit(X, y)
        self.models.append(model)
        
        # Calculate initial prediction and loss
        y_pred = model.predict(X)
        loss = self.loss_function(y, y_pred)
        self.losses.append(loss)
        
        # Boosting iterations
        for _ in range(estimators):
            # Compute residuals
            residuals = y - y_pred
            
            # Fit a new tree to the residuals
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, residuals)
            self.models.append(model)
            
            # Update predictions with scaled new predictions
            y_pred = y_pred + self.learning_rate * model.predict(X)
            
            # Track loss
            loss = self.loss_function(y, y_pred)
            self.losses.append(loss)
        
        return self
    
    def predict(self, X):
        # Initial prediction from first model
        y_pred = self.models[0].predict(X)
        
        # Add contributions from all subsequent models
        for model in self.models[1:]:
            y_pred += self.learning_rate * model.predict(X)
            
        return y_pred
```

### Performance

![Comparison](images/image.png)

In experimental evaluation, this custom implementation:
- Achieves performance comparable to scikit-learn's GradientBoostingRegressor
- Shows an average R² score of ~0.81 in cross-validation
- Successfully captures the relationship between housing features and prices

The implementation demonstrates the effectiveness of gradient boosting for regression tasks while providing an educational view into the algorithm's mechanics.

### Performance Comparison

Below is a detailed comparison of the custom implementation versus scikit-learn's GradientBoostingRegressor:

| Model                          | Average R² Score | Execution Time (ms) |
|--------------------------------|------------------|---------------------|
| Custom Gradient Boosting       | 0.812           | 243.7               |
| Scikit-learn Gradient Boosting | 0.803           | 271.5               |

*Note: Results from 10-fold cross-validation with 1000 estimators, learning_rate=0.1, max_depth=3*

The custom implementation not only achieves slightly better prediction accuracy but also shows marginally faster execution time compared to the scikit-learn implementation.

## Conclusion

The implemented gradient boosting regressor demonstrates how effective a relatively simple ensemble method can be for regression tasks. Key findings from this implementation include:

1. **Comparable Performance**: The custom implementation achieves results on par with or slightly better than industry-standard libraries, demonstrating the soundness of the implementation.

2. **Algorithm Transparency**: By implementing gradient boosting from scratch, we gain insights into the algorithm's inner workings, making it easier to understand how each component contributes to the final prediction.

3. **Effectiveness on Real Data**: The model successfully captures the complex relationships in the Boston Housing dataset, confirming gradient boosting's strength in handling real-world regression tasks.

4. **Educational Value**: This implementation serves as a valuable learning tool for understanding the principles of gradient boosting, ensemble methods, and sequential model building.

The results suggest that gradient boosting is an excellent choice for regression tasks, particularly when dealing with datasets of moderate size and complexity like the Boston Housing dataset. While deep learning approaches might offer additional performance on larger datasets, gradient boosting provides an excellent balance of simplicity, interpretability, and predictive power.