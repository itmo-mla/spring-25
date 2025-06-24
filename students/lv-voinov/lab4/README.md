# Лабораторная работа 4

## Датасет

20 newsgroups

<https://www.kaggle.com/datasets/crawford/20-newsgroups>

Датасет для работы с текстом по категориям.

## Латентное размещение Дирихле

Латентное размещение Дирихле (LDA) — это байесовская модель для автоматического выявления скрытых тематических структур в текстовых коллекциях. Она представляет документы как смеси тем, а темы — как распределения слов.

Генеративный процесс LDA предполагает, что каждый документ формируется путем выбора распределения тем из распределения Дирихле с параметром α. Для каждого слова в документе сначала выбирается тема из этого распределения, а затем конкретное слово из распределения слов темы, которое также задается распределением Дирихле с параметром β.

Для обучения модели по существующим документам применяется байесовский вывод. Чаще всего, в том числе в этой работе, применяется сэмплирование по Гиббсу. Этот итеративный алгоритм последовательно пересматривает тематические назначения слов, используя условные вероятности, основанные на текущем распределении тем в документе и слов в темах. После сходимости вычисляются итоговые параметры: распределение слов по темам и распределение тем по документам.

Качество модели оценивается через когерентность тем, измеряющую семантическую согласованность топ-слов на основе их совместной встречаемости. LDA широко применяется для анализа текстов, тематического моделирования и уменьшения размерности данных.

### Код алгоритма

```python
class LDAGibbs:
    def __init__(self, n_topics=10, alpha=0.1, beta=0.1, max_iter=100):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.vocab = None
        self.phi = None
        self.theta = None
        
    def fit(self, X, vocab):
        n_docs = X.shape[0]
        self.vocab = vocab
        self.vocab_size = len(vocab)
        
        self.n_dk = np.zeros((n_docs, self.n_topics)) + self.alpha
        self.n_kw = np.zeros((self.n_topics, self.vocab_size)) + self.beta
        self.n_k = np.zeros(self.n_topics) + self.vocab_size * self.beta
        
        self.z = []
        for d in range(n_docs):
            doc = X[d].tocoo()
            cols = doc.col
            data = doc.data
            word_indices = np.repeat(cols, data)
            n_words_in_doc = len(word_indices)
            
            topics_in_doc = np.random.choice(self.n_topics, size=n_words_in_doc)
            self.z.append(topics_in_doc)
            
            for w, topic in zip(word_indices, topics_in_doc):
                self.n_dk[d, topic] += 1
                self.n_kw[topic, w] += 1
                self.n_k[topic] += 1
        
        for iteration in range(self.max_iter):
            for d in range(n_docs):
                doc = X[d].tocoo()
                cols = doc.col
                data = doc.data
                word_indices = np.repeat(cols, data)
                topics_in_doc = self.z[d]
                
                for i in range(len(word_indices)):
                    w = word_indices[i]
                    old_topic = topics_in_doc[i]
                    
                    self.n_dk[d, old_topic] -= 1
                    self.n_kw[old_topic, w] -= 1
                    self.n_k[old_topic] -= 1
                    
                    p_topics = (self.n_dk[d] / (self.n_dk[d].sum() + self.n_topics * self.alpha)) * \
                               (self.n_kw[:, w] / (self.n_k + 1e-12))
                    p_topics = p_topics / p_topics.sum()
                    
                    new_topic = np.random.choice(self.n_topics, p=p_topics)
                    topics_in_doc[i] = new_topic
                    
                    self.n_dk[d, new_topic] += 1
                    self.n_kw[new_topic, w] += 1
                    self.n_k[new_topic] += 1
        
        self.phi = self.n_kw / self.n_k[:, np.newaxis]
        self.theta = (self.n_dk) / (self.n_dk.sum(axis=1)[:, np.newaxis] + 1e-12)
        return self
    
    def get_topics(self, n_words=10):
        topic_words = []
        for k in range(self.n_topics):
            top_indices = self.phi[k].argsort()[-n_words:][::-1]
            topic_words.append([self.vocab[i] for i in top_indices])
        return topic_words
```

### Результаты работы

Было проведено обучение LDA на подмножестве датасета 20 newsgroups с 10 темами.

Результаты ручного алгоритма:

```
Ручной LDA:
  Время обучения: 51.58 сек
  Когерентность тем: -2.1486
Тема 1: entry, output, use, program, file
Тема 2: god, people, don, say, think
Тема 3: right, know, like, power, think
Тема 4: server, work, using, time, use
Тема 5: file, number, program, information, oname
Тема 6: cancer, people, group, book, just
Тема 7: car, think, game, vitamin, good
Тема 8: edu, keyboard, pc, available, xfree86
Тема 9: 00, 10, government, new, 20
Тема 10: like, want, know, just, way
```

Результаты библиотечного алгоритма:

```
Sklearn LDA:
  Время обучения: 5.61 сек
  Когерентность тем: -2.4519
Тема 1: entry, use, like, xfree86, file
Тема 2: output, file, government, people, like
Тема 3: cancer, clutch, people, hiv, information
Тема 4: 00, 10, 50, 1st, 15
Тема 5: like, know, people, does, just
Тема 6: god, know, want, don, think
Тема 7: keyboard, like, pc, new, price
Тема 8: gm, game, 03, team, 02
Тема 9: 10, 00, people, right, don
Тема 10: vitamin, retinol, use, liver, time
```

### Выводы

Библиотечный алгоритм работает в 10 раз быстрее ручного, но когерентность UMass чуть больше у ручного.
