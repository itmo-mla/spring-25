import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


def metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Confusion matrix: {confusion}")
    print(f"Classification report: {classification_report(y_true, y_pred)}")
    return accuracy, f1, precision, recall, confusion
    


def gaussian_pdf(x, mean, variance):
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-(x - mean) ** 2 / (2 * variance))


class NaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        self.variance = np.zeros((len(self.classes), X.shape[1]))
        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.prior = np.zeros(len(self.classes))
        for i, class_ in enumerate(self.classes):   
            X_class = X[y == class_]
            self.variance[i] = np.var(X_class, axis=0) + self.var_smoothing
            self.mean[i] = np.mean(X_class, axis=0)
            self.prior[i] = len(X_class) / len(y)
    
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y_pred = []
        for x in X:
            posteriors = []
            for i, _ in enumerate(self.classes):
                prior = np.log(self.prior[i])
                class_conditional = np.sum(np.log(gaussian_pdf(x, self.mean[i], self.variance[i])))
                posterior = prior + class_conditional
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)
    
    

