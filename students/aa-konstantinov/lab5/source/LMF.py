import numpy as np 
from typing import Optional, Sequence, Tuple, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
RatingTriplet = Tuple[int, int, float]
class LMF:
    """SGD LMF"""
    def __init__(self, n_users, n_items, n_factors, use_bias=True):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias_u = np.zeros(n_users, dtype=np.float32)
            self.bias_i = np.zeros(n_items, dtype=np.float32)
            self.bias_global = 0.0

    def fit(self, X, lr=0.01, reg=0.01, n_epochs=10, verbose=True):
        
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.verbose = verbose
        
        if self.use_bias:
            ratings = [r for _, _, r in X]
            self.bias_global = np.mean(ratings)
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(X)
            epoch_loss = 0.0
            
            for u, i, r in X:
                u = int(u)
                i = int(i)
                r_hat = self.predict(u, i, add_bias=True)
                err = r - r_hat
                epoch_loss += err ** 2
                
                if self.use_bias:
                    self.bias_u[u] += self.lr * (err - self.reg * self.bias_u[u])
                    self.bias_i[i] += self.lr * (err - self.reg * self.bias_i[i])    
                p_u = self.P[u].copy()
                q_i = self.Q[i].copy()
            
                self.P[u] += self.lr * (err * q_i - self.reg * p_u)
                self.Q[i] += self.lr * (err * p_u - self.reg * q_i)
            
            epoch_rmse = np.sqrt(epoch_loss / len(X))
            
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.n_epochs}  RMSE={epoch_rmse:.4f}")
            
            if epoch_rmse < 0.01:
                if self.verbose:
                    print("Достигнута достаточная точность. Останавливаем обучение.")
                break
                
        return self
    
    def predict(self, user: int, item: int, add_bias: bool = True) -> float:
        """Predict rating / interaction strength for a (user, item) pair."""
        score = float(np.dot(self.P[user], self.Q[item]))
        if add_bias and self.use_bias:
            score += self.bias_global + self.bias_u[user] + self.bias_i[item]
        return score
    
    def rmse(self, interactions) -> float:
        """Compute RMSE on supplied triples."""
        data = self._prepare_data(interactions)
        predictions = []
        actuals = []
        for u, i, r in data:
            predictions.append(self.predict(int(u), int(i), add_bias=True))
            actuals.append(r)
        return float(np.sqrt(mean_squared_error(actuals, predictions)))
    
    @staticmethod
    def _prepare_data(interactions: Sequence[RatingTriplet]) -> np.ndarray:
        arr = np.asarray(interactions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("interactions must be array-like (n_samples,3)")
        return arr
            
        

if __name__ == "__main__":
    import time
    
    DATA_PATH = "/Users/a.konstantinov/Documents/less/spring-25/students/aa-konstantinov/lab5/source/ml-100k/u.data"      
    
    data = pd.read_csv(DATA_PATH,
                   sep="\t", names=["user","item","rating","ts"])
    
    data = data[data.columns[0:3]]
    data["user"] -= 1              
    data["item"] -= 1

    train, temp = train_test_split(data.values, test_size=0.20, random_state=42)
    val,   test = train_test_split(temp,       test_size=0.50, random_state=42)

    n_users = np.max(data['user']) + 1
    n_items = np.max(data['item']) + 1
    n_factors = 40
    
    print("=" * 50)
    print("Обучение LMF")
    print("=" * 50)
    start = time.time()
    model = LMF(n_users, n_items, n_factors)
    model.fit(train, lr=0.01, reg=0.05, n_epochs=25, verbose=True)
    end = time.time()
    print(f"Time: {end - start} seconds")

    print("Val RMSE:", model.rmse(val))
    print("Test RMSE:", model.rmse(test))


    print("=" * 50)
    print("Обучение SVD из Surprise")
    print("=" * 50)
    start = time.time()
    from surprise import Dataset, Reader, SVD, accuracy
    from surprise.model_selection import train_test_split 
    reader = Reader(rating_scale=(1, 5))
    
    train_df = pd.DataFrame(train, columns=['user', 'item', 'rating'])
    val_df = pd.DataFrame(val, columns=['user', 'item', 'rating'])
    test_df = pd.DataFrame(test, columns=['user', 'item', 'rating'])
    
    train_df['user'] += 1
    train_df['item'] += 1
    val_df['user'] += 1
    val_df['item'] += 1
    test_df['user'] += 1
    test_df['item'] += 1

    trainset = Dataset.load_from_df(train_df, reader).build_full_trainset()
    
    testset = [(int(uid), int(iid), float(r)) for uid, iid, r in test_df.values]
    valset = [(int(uid), int(iid), float(r)) for uid, iid, r in val_df.values]
    
    algo = SVD(
        n_factors=40,          
        n_epochs=25,
        lr_all=0.01,
        reg_all=0.05,          
        random_state=0,
        biased=True            
    )
    algo.fit(trainset)
    end = time.time()
    print(f"Time: {end - start} seconds")
    preds_test = algo.test(testset)
    preds_val = algo.test(valset)

    print("Surprise-SVD Val RMSE =", accuracy.rmse(preds_val, verbose=False))
    print("Surprise-SVD Test RMSE =", accuracy.rmse(preds_test, verbose=False))
    



    
    
    


           