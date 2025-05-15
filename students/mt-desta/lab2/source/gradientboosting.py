from sklearn.tree import DecisionTreeRegressor

class GradientBoosting:
    def __init__(self, learning_rate=0.1, max_depth=3, loss_func=None):
        self.models = []
        self.losses = []

        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.loss_function = loss_func
    
    def fit(self, X, y, estimators=1000):
        # Initialize the model with the first tree      
        model = DecisionTreeRegressor(max_depth=self.max_depth)
        model.fit(X, y)
        self.models.append(model)

        # Calculate the initial loss
        y_pred = model.predict(X)
        loss = self.loss_function(y, y_pred)
        self.losses.append(loss)

        # Fit the model for num_iterations
        for _ in range(estimators):
            # Compute residuals
            residuals = y - y_pred

            # Fit a new tree to the negative gradient   
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, residuals)
            self.models.append(model)

            # Update the predictions
            y_pred = y_pred + self.learning_rate * model.predict(X) 

            # Calculate the loss
            loss = self.loss_function(y, y_pred)
            self.losses.append(loss)
        
        return self
    
    def predict(self, X):
        y_pred = self.models[0].predict(X)

        for model in self.models[1:]:
            y_pred += self.learning_rate * model.predict(X)
        return y_pred
    
    
    
    

