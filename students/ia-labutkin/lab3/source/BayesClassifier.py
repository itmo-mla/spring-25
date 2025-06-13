import numpy as np
from sklearn.model_selection import KFold


def count_accuracy(y_pred, y_real): #Подсчёт точности классификации
    return np.sum(y_pred==y_real)/len(y_real)

def cross_validation(alghorithm, X,y,n_splits):#Кросс-валидация
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs=[]
    for trains,tests in kf.split(X):#n разбиений выборки на test и train
        alghorithm.fit(X[trains,:],y[trains])
        accs.append(count_accuracy(alghorithm.predict(X[tests,:]),y[tests]))#Оценивание на каждом разбиении
    return np.mean(accs)#Выдача среднего accuracy по каждому разбиению

class GaussianNBC():
    def __init__(self):
        pass
    def fit(self, X_train,y_train):
        vals, counts=np.unique(y_train, return_counts=True) #Получение всех значений и частоты их попадания
        self.params={}#Словарь значение: его средние, стандартные отклонения и вероятности
        for i, value in enumerate(vals):#Каждое из значений добавляем в словарь
            self.params[value]=(np.mean(X_train[np.where(y_train==value)], axis=0), np.std(X_train[np.where(y_train==value)], axis=0), np.log(counts[i])/np.sum(counts))
    def predict(self, X_test):
        probabilities=[]#Массив с формулами по каждому из значений
        for target in self.params.keys():
            params=self.params[target]#Получаем параметры по каждому значению
            probability=params[2]-np.sum(((X_test-params[0])**2)/(2*params[1]**2)+np.log(params[1]),axis=1)#Расчёт по формуле
            probabilities.append(probability)#Добавление результатов
        return np.array(list(self.params.keys()))[np.argmax(np.array(probabilities), axis=0)]#Возвращается класс, соответствующий индексу аргмакса результата по формуле