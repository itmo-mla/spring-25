import numpy as np

class CategoricalNB:
    def fit(self,X,y):
        self.classes = np.unique(y)
        self.features = X.shape[1]

        self.categorical_features = [None]*self.features

        for f in range(self.features):
            self.categorical_features[f] = np.unique(X[:,f])

        self.priors = np.zeros(len(self.classes))

        for c in self.classes:
            self.priors[c] = np.sum(y == c) / len(y)

        self.conditional_probabilities = {}

        for f in range(self.features):
            self.conditional_probabilities[f] = {}
            for v in self.categorical_features[f]:
                self.conditional_probabilities[f][v] = {}
                for c in self.classes:
                    # Initialize with a placeholder value
                    self.conditional_probabilities[f][v][c] = 0

        for f in range(self.features):
            for v in self.categorical_features[f]:
                for c in self.classes:
                    count = np.sum((X[:,f] == v) & (y == c))
                    total = np.sum(y == c)
                    num_values = len(self.categorical_features[f])

                    self.conditional_probabilities[f][v][c] = (count + 1) / (total + num_values)


    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.zeros(num_samples, dtype=int)
        
        # Process each sample individually
        for i in range(num_samples):
            sample = X[i]
            
            # Calculate posterior probabilities for each class
            posteriors = {}
            for c in self.classes:
                # Start with log prior
                posterior = np.log(self.priors[c])
                
                # Add log likelihoods for each feature
                for f in range(self.features):
                    feature_value = sample[f]
                    
                    # Handle unseen feature values
                    if feature_value in self.conditional_probabilities[f]:
                        posterior += np.log(self.conditional_probabilities[f][feature_value][c])
                    else:
                        # Use Laplace smoothing for unseen values
                        num_values = len(self.categorical_features[f])
                        posterior += np.log(1.0 / num_values)
                
                posteriors[c] = posterior
            
            # Find class with highest posterior probability
            predictions[i] = max(posteriors, key=posteriors.get)
        
        return predictions 
            
            
