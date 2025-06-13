import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error



def prepare_data(df):
    unique_users = sorted(df['userId'].unique())
    unique_items = sorted(df['movieId'].unique())

    user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
    item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}

    df['user_idx'] = df['userId'].map(user_id_map)
    df['item_idx'] = df['movieId'].map(item_id_map)


    R= df[['user_idx', 'item_idx', 'rating']].to_numpy()
    R_train, R_test = train_test_split(R, test_size=0.2, random_state=42)
    return R_train, R_test, len(user_id_map), len(item_id_map)

class LMF:
    def __init__(self, n_users, n_movies, n_factors):
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.1, (n_movies, n_factors))
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_movies, dtype=np.float32)
    def fit(self, R,  n_iters=100, alpha=0.01, lam=0.1, eps=0.001):
        losses=[]
        for _ in range(n_iters):
            np.random.shuffle(R)
            mse=0
            avg_loss=99999
            for d in R:
                u,i,r=d.astype(int)
                pred = self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                eui = r - pred
                self.b_u[u] += alpha * (eui - lam * self.b_u[u])
                self.b_i[i] += alpha * (eui - lam * self.b_i[i])
                self.P[u] += alpha * (eui * self.Q[i] - lam * self.P[u])
                self.Q[i] += alpha * (eui * self.P[u] - lam * self.Q[i])
                mse+=eui**2
            avg_loss_new=mse/R.shape[0]
            if abs(avg_loss_new-avg_loss)<eps:
                break
            else:
                avg_loss=avg_loss_new
                losses.append(avg_loss) 
        return losses
    def predict(self, R):
        y_pred=np.array([])
        for d in R:
            u,i,_=d.astype(int)
            y_pred=np.append(y_pred,self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i]))
        return y_pred
    
ratings = pd.read_csv("ml-latest-small/ratings.csv")[['userId','movieId','rating']]
ratings['userId']-=1
ratings['movieId']-=1

R_train, R_test, n_users, n_movies=prepare_data(ratings)
lmf=LMF(n_users, n_movies, 20)
losses=lmf.fit(R_train,n_iters=50)
y_pred=lmf.predict(R_test)
y_real=R_test[:,2]
plt.plot([i for i in range(50)], losses)
plt.title('График обучения')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.show()
print(f"RMSE: {np.sqrt(mean_squared_error(y_real, y_pred))}")
print(f"MAE: {mean_absolute_error(y_real, y_pred)}")


