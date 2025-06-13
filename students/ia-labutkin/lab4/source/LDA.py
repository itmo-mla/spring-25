import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

def get_topics_words(phi, feature_names, top_n=10):#Получение топ-слов по каждой теме
    return [
        [feature_names[i] for i in topic.argsort()[-top_n:][::-1]]
        for topic in phi
    ]

def get_coherence_score(vectorizer, phi, texts, metric='u_mass'):#Расчёт когерентности тем

    feature_names = vectorizer.get_feature_names_out()
    topics = get_topics_words(phi, feature_names, top_n=10)#Получение топ-слов
    analyzer = vectorizer.build_analyzer()
    tokens = [analyzer(doc) for doc in texts]#Получение токенов
    dictionary = Dictionary(tokens)#Получение словаря

    
    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokens,
        dictionary=dictionary,
        coherence=metric
    )
    coherence_score = coherence_model.get_coherence()#Расчёт когерентности
    print(f"Когерентность тем ({metric}): {coherence_score:.4f}")



class LDA():
    def __init__(self, num_themes, alpha=1, beta=1):
        self.num_themes=num_themes#Число тем
        self.alpha=alpha#Коэффициент регуляризации по theta
        self.beta=beta#Коэффициент регуляризации по phi
        self.vectorizer=CountVectorizer()#Векторизатор (BoW)

    def fit_predict(self,train_text, max_steps=100, eps_theta=1e-6,eps_phi=1e-8):#Получение и матрицы phi, и матрицы theta
        X_train = self.vectorizer.fit_transform(train_text)#Векторизация текста
        num_docs, num_words=X_train.shape
        #Инициализация и приведение в необходимые условия матриц phi и theta
        theta = np.random.rand(self.num_themes, num_docs)
        theta /= np.sum(theta, axis=0, keepdims=True)
        phi = np.random.rand(self.num_themes, num_words)
        phi /= np.sum(phi, axis=1, keepdims=True)
        for _ in range(max_steps):
            #E-шаг
            standartizer = np.clip(np.matmul(theta.T, phi), 1e-12, None)#Знаменатель, по которому идёт нормализация
            ptdw=np.zeros((self.num_themes,num_docs,num_words))
            for i in range(self.num_themes):#Расчёт числителя и стандартизация
                ptdw[i]=np.outer(theta[i], phi[i])/standartizer
            #M-шаг
            nwt = np.zeros((self.num_themes, num_words))
            ntd = np.zeros((self.num_themes, num_docs))
            for i in range(self.num_themes):
                for d in range(num_docs):
                    row = X_train[d].toarray().ravel() 
                    nwt[i] += ptdw[i, d] * row
                    ntd[i, d] = np.dot(ptdw[i, d], row)
            phi_new=(nwt+self.beta-1)/np.sum(nwt+self.beta-1,axis=1,keepdims=True)#Расчёт нового phi
            theta_new=(ntd+self.alpha-1)/np.sum(ntd+self.alpha-1,axis=0,keepdims=True)#Расчёт нового theta
            diff_phi=np.mean(np.abs(phi-phi_new))
            diff_theta=np.mean(np.abs(theta-theta_new))<=eps_theta
            theta=theta_new
            phi=phi_new
            if diff_phi<=eps_phi and diff_theta<=eps_theta:#Если есть сходимость, то раньше выходим из цикла
                break
        self.phi=phi
        return phi,theta
    
    def predict(self,text, max_steps=100, eps=1e-6):#Получение theta при уже заданном phi
        #То же самое, что и в предыдущей функции, но без обновления матрицы phi
        X = self.vectorizer.transform(text)
        num_themes, num_words = self.phi.shape
        num_docs=X.shape[0]
        theta = np.random.rand(num_themes, X.shape[0])
        theta /= np.sum(theta, axis=0, keepdims=True)
        for _ in range(max_steps):
            #E-шаг
            standartizer = np.clip(np.matmul(theta.T, self.phi), 1e-12, None)
            ptdw=np.zeros((num_themes,num_docs,num_words))
            for i in range(num_themes):
                ptdw[i]=np.outer(theta[i], self.phi[i])/standartizer
            #M-шаг
            ntd = np.zeros((num_themes, num_docs))
            for i in range(num_themes):
                for d in range(num_docs):
                    row = X[d].toarray().ravel() 
                    ntd[i, d] = np.dot(ptdw[i, d], row)
            theta_new=(ntd+self.alpha-1)/np.sum(ntd+self.alpha-1,axis=0,keepdims=True)
            theta=theta_new
            diff=np.mean(np.abs(theta-theta_new))
            if diff<=eps:
                break
        return theta
