from read import read_data
from bagging import BaggingClassificator
from metrics import all_metrics
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X, y = read_data('data/train.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    models = [
        BaggingClassificator(n_estimators=100, max_samples=0.8, random_state=42),
        RandomForestClassifier(n_estimators=100, max_samples=0.8, random_state=42)
    ]

    metrics = []
    times = []
    for model in models:
        start_time = time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        times.append(time() - start_time)
        metrics.append(all_metrics(y_test, y_pred))

    print(f"My time: {times[0]}")
    print(f"My Accuracy: {metrics[0][0]}")
    print(f"My Recall: {metrics[0][1]}")
    print(f"My Precision: {metrics[0][2]}")
    print(f"My f1: {metrics[0][3]}")

    print()

    print(f"Sklearn time: {times[1]}")
    print(f"Sklearn Accuracy: {metrics[1][0]}")
    print(f"Sklearn Recall: {metrics[1][1]}")
    print(f"Sklearn Precision: {metrics[1][2]}")
    print(f"Sklearn f1: {metrics[1][3]}")