import numpy as np
import pandas as pd
import urllib.request
import os
import time
from typing import Optional, Sequence, Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_train_test_split
from tabulate import tabulate

RatingTriplet = Tuple[int, int, float]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    'ml-100k': {
        'url': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
        'file': os.path.join(SCRIPT_DIR, 'ml-100k/u.data'),
        'sep': '\t',
        'names': ['user', 'item', 'rating', 'ts'],
        'rating_scale': (1, 5)
    },
    'ml-1m': {
        'url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'file': os.path.join(SCRIPT_DIR, 'ml-1m/ratings.dat'),
        'sep': '::',
        'names': ['user', 'item', 'rating', 'ts'],
        'rating_scale': (1, 5)
    }
}

def download_dataset(dataset_name: str):
    dataset_info = DATASETS[dataset_name]
    local_path = os.path.join(SCRIPT_DIR, f"{dataset_name}.zip")
    extract_dir = os.path.join(SCRIPT_DIR, dataset_info['file'].split('/')[0])
    
    if not os.path.exists(extract_dir):
        urllib.request.urlretrieve(dataset_info['url'], local_path)
        
        import zipfile
        with zipfile.ZipFile(local_path, 'r') as zip_ref:
            zip_ref.extractall(SCRIPT_DIR)
        
        os.remove(local_path)
        print(f"{dataset_name} dataset downloaded")

def normalize_user_ratings(data: pd.DataFrame) -> pd.DataFrame:
    user_stats = data.groupby('user')['rating'].agg(['min', 'max'])
    normalized_data = data.copy()
    
    normalized_data['rating'] = normalized_data['rating'].astype(np.float64)
    
    for user in user_stats.index:
        min_rating = user_stats.loc[user, 'min']
        max_rating = user_stats.loc[user, 'max']
        
        if min_rating == max_rating:
            continue
            
        mask = normalized_data['user'] == user
        ratings = normalized_data.loc[mask, 'rating']
        
        normalized_ratings = 1 + 4 * (ratings - min_rating) / (max_rating - min_rating)
        normalized_data.loc[mask, 'rating'] = normalized_ratings
    
    return normalized_data

def load_dataset(dataset_name: str) -> pd.DataFrame:
    dataset_info = DATASETS[dataset_name]
    
    data = pd.read_csv(dataset_info['file'],
                      sep=dataset_info['sep'],
                      names=dataset_info['names'],
                      engine='python')
    
    data = data[['user', 'item', 'rating']]
    data['user'] = data['user'].astype('category').cat.codes
    data['item'] = data['item'].astype('category').cat.codes
    
    data = normalize_user_ratings(data)
    
    data['rating'] = data['rating'].round(2)
    
    return data

class LatentFactorModel:
    def __init__(self, n_users: int, n_items: int, n_factors: int = 40):
        self.user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = 0.0

    def _predict_single(self, user_idx: int, item_idx: int) -> float:
        return (self.global_bias + 
                self.user_bias[user_idx] + 
                self.item_bias[item_idx] + 
                np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))

    def _update_factors(self, user_idx: int, item_idx: int, 
                       rating: float, learning_rate: float, reg: float):
        prediction = self._predict_single(user_idx, item_idx)
        error = rating - prediction

        self.user_bias[user_idx] += learning_rate * (error - reg * self.user_bias[user_idx])
        self.item_bias[item_idx] += learning_rate * (error - reg * self.item_bias[item_idx])

        user_factor = self.user_factors[user_idx].copy()
        item_factor = self.item_factors[item_idx].copy()

        self.user_factors[user_idx] += learning_rate * (error * item_factor - reg * user_factor)
        self.item_factors[item_idx] += learning_rate * (error * user_factor - reg * item_factor)

    def fit(self, ratings: np.ndarray, learning_rate: float = 0.01, 
            reg: float = 0.05, n_epochs: int = 15) -> 'LatentFactorModel':
        self.global_bias = np.mean([r for _, _, r in ratings])

        for epoch in range(n_epochs):
            np.random.shuffle(ratings)
            epoch_loss = 0.0

            for user_idx, item_idx, rating in ratings:
                user_idx, item_idx = int(user_idx), int(item_idx)
                self._update_factors(user_idx, item_idx, rating, learning_rate, reg)
                prediction = self._predict_single(user_idx, item_idx)
                epoch_loss += (rating - prediction) ** 2

            rmse = np.sqrt(epoch_loss / len(ratings))
            print(f"Epoch {epoch + 1}/{n_epochs}  Train RMSE={rmse:.4f}")

        return self

    def evaluate(self, test_ratings: np.ndarray) -> Tuple[float, float]:
        predictions = []
        actuals = []
        
        for user_idx, item_idx, rating in test_ratings:
            user_idx, item_idx = int(user_idx), int(item_idx)
            predictions.append(self._predict_single(user_idx, item_idx))
            actuals.append(rating)
            
        rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
        mae = float(mean_absolute_error(actuals, predictions))
        
        return rmse, mae

def evaluate_model(dataset_name: str) -> Dict:
    download_dataset(dataset_name)
    data = load_dataset(dataset_name)
    
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*50}")
    print(f"Number of users: {data['user'].nunique()}")
    print(f"Number of items: {data['item'].nunique()}")
    print(f"Number of ratings: {len(data)}")

    train, test = train_test_split(data.values, test_size=0.20, random_state=33)
    
    n_users = data['user'].nunique()
    n_items = data['item'].nunique()
    n_factors = 40
    
    start_time = time.time()
    model = LatentFactorModel(n_users, n_items, n_factors)
    model.fit(train, learning_rate=0.01, reg=0.05, n_epochs=15)
    custom_time = time.time() - start_time
    
    test_rmse, test_mae = model.evaluate(test)
    
    reader = Reader(rating_scale=DATASETS[dataset_name]['rating_scale'])
    train_df = pd.DataFrame(train, columns=['user', 'item', 'rating'])
    test_df = pd.DataFrame(test, columns=['user', 'item', 'rating'])
    
    for df in [train_df, test_df]:
        df[['user', 'item']] += 1
    
    trainset = Dataset.load_from_df(train_df, reader).build_full_trainset()
    testset = [(int(uid), int(iid), float(r)) for uid, iid, r in test_df.values]
    
    start_time = time.time()
    algo = SVD(n_factors=40, n_epochs=15, lr_all=0.01, reg_all=0.05, random_state=33)
    algo.fit(trainset)
    surprise_time = time.time() - start_time
    
    preds_test = algo.test(testset)
    surprise_test_rmse = accuracy.rmse(preds_test, verbose=False)
    surprise_test_mae = accuracy.mae(preds_test, verbose=False)

    print(f"\nResults for {dataset_name}:")
    table_data = [
        ['Custom', f"{custom_time:.2f}", f"{test_rmse:.4f}", f"{test_mae:.4f}"],
        ['Surprise-SVD', f"{surprise_time:.2f}", f"{surprise_test_rmse:.4f}", f"{surprise_test_mae:.4f}"]
    ]
    headers = ['Model', 'Time (s)', 'RMSE', 'MAE']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    return {
        'dataset': dataset_name,
        'n_users': n_users,
        'n_items': n_items,
        'n_ratings': len(data),
        'custom_time': custom_time,
        'custom_test_rmse': test_rmse,
        'custom_test_mae': test_mae,
        'surprise_time': surprise_time,
        'surprise_test_rmse': surprise_test_rmse,
        'surprise_test_mae': surprise_test_mae
    }

def main():
    results = []

    for dataset_name in DATASETS.keys():
        try:
            result = evaluate_model(dataset_name)
            results.append(result)
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")

    print("\n" + "="*100)

    table_data = []
    for r in results:
        table_data.append([
            r['dataset'],
            f"{r['n_users']:,}",
            f"{r['n_items']:,}",
            f"{r['n_ratings']:,}",
            f"{r['custom_time']:.2f}",
            f"{r['custom_test_rmse']:.4f}",
            f"{r['custom_test_mae']:.4f}",
            f"{r['surprise_time']:.2f}",
            f"{r['surprise_test_rmse']:.4f}",
            f"{r['surprise_test_mae']:.4f}"
        ])
    
    headers = [
        'Dataset', 'Users', 'Items', 'Ratings',
        'Custom Time', 'Custom RMSE', 'Custom MAE',
        'Surprise Time', 'Surprise RMSE', 'Surprise MAE'
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

if __name__ == "__main__":
    main()



