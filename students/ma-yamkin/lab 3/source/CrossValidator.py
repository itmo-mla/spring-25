import numpy as np


class CrossValidation:
    def __init__(self, n_splits, random_state=12, shuffle=True):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

    def split(self, X):
        X = np.array(X)
        n_samples = len(X)

        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.seed(self.random_state)
            indices = np.random.permutation(indices)

        fold_size = n_samples // self.n_splits
        folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(self.n_splits)]

        remainder = n_samples % self.n_splits
        if remainder != 0:
            folds[-1] = np.concatenate([folds[-1], indices[-remainder:]])

        for fold_idx in range(self.n_splits):
            val_indices = folds[fold_idx]
            train_indices = np.concatenate([f for i, f in enumerate(folds) if i != fold_idx])
            yield train_indices, val_indices

    def eval(self, X, y, model, metric):
        metrics = []
        X, y = np.array(X), np.array(y)

        for train_idx, val_idx in self.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, y_valid = X[val_idx], y[val_idx]

            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            try:
                metrics.append(metric(pred, y_valid))
            except ValueError:
                metrics.append(metric(pred, y_valid, average='macro'))

        return np.array(metrics)
