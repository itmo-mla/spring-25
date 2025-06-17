## 1. Теория алгоритма Latent Dirichlet Allocation

Latent Dirichlet Allocation (LDA) — это генеративная вероятностная модель корпуса документов, позволяющая выявлять скрытые (латентные) темы. Основная идея:

1. **Генеративный процесс** для корпуса из \$D\$ документов, словарь размера \$V\$ и \$K\$ тем:

   1. Для каждой темы \$k = 1, \dots, K\$ генерируется вектор \$\beta\_k \sim \operatorname{Dirichlet}(\eta)\$, задающий распределение по словам.
   2. Для каждого документа \$d = 1, \dots, D\$ генерируется вектор \$\theta\_d \sim \operatorname{Dirichlet}(\alpha)\$, задающий распределение тем внутри документа.
   3. Для каждого слова \$n = 1, \dots, N\_d\$ в документе:

      $$
        z_{dn} \mid \theta_d \sim \operatorname{Categorical}(\theta_d), 
        \quad
        w_{dn} \mid z_{dn}, \beta \sim \operatorname{Categorical}(\beta_{z_{dn}}).
      $$

2. **Вариационный вывод**
   Точное апостериорное распределение \$p(\beta, \theta, z \mid w)\$ вычислить невозможно. Вводим вариационное приближение

   $$
     q(\beta, \theta, z)
     = \prod_{k=1}^K q(\beta_k \mid \lambda_k)\;\prod_{d=1}^D \Bigl[q(\theta_d \mid \gamma_d)\,\prod_{n=1}^{N_d}q(z_{dn} \mid \phi_{dn})\Bigr],
   $$

   где

   * \$q(\beta\_k) = \operatorname{Dirichlet}(\lambda\_k)\$,
   * \$q(\theta\_d) = \operatorname{Dirichlet}(\gamma\_d)\$,
   * \$q(z\_{dn}) = \operatorname{Categorical}(\phi\_{dn})\$.

   Оптимизируем вариационные параметры \${\lambda\_k}\$ (глобальные) и \${\gamma\_d, \phi\_{dn}}\$ (локальные) путём максимизации ELBO, чередуя:

   * **E-шаг**: для каждого документа \$d\$ фиксируем \$\lambda\$ и обновляем \${\gamma\_d, \phi\_{dn}}\$ через координатный спуск:

     $$
       \begin{aligned}
       \phi_{dv,k} &\propto \exp\bigl(\mathbb{E}[\log\theta_{dk}]\bigr)\,\exp\bigl(\mathbb{E}[\log\beta_{kv}]\bigr),\\
       \gamma_{dk} &= \alpha_k + \sum_{v=1}^V x_{dv}\,\phi_{dv,k}.
       \end{aligned}
     $$
   * **M-шаг**: аккумулируем «мягкие» счёты

     $$
       s_{kv} = \sum_{d=1}^D \sum_{v=1}^V x_{dv}\,\phi_{dv,k},
       \quad
       \lambda_{kv} = \eta + s_{kv},
     $$

     и пересчитываем \$\exp\bigl(\mathbb{E}\[\log\beta\_{kv}]\bigr)\$.

3. **Критерии остановки**

   * Сходимость по ELBO или изменение параметров ниже порога \$\mathrm{tol}\$.
   * Достижение максимального числа итераций \$\mathrm{max\_iter}\$.

---

## 2. Описание датасета

Используется часть корпуса 20 Newsgroups из шести категорий:

* **Всего документов**: 3 402

* **Распределение по классам**:

  * sci.space: 480
  * comp.graphics: 584
  * soc.religion.christian: 600
  * alt.atheism: 593
  * talk.politics.guns: 599
  * rec.sport.hockey: 546

* **Средняя длина документов** (в словах):

  |          Класс         | Среднее | Медиана |
  | :--------------------: | :-----: | :-----: |
  |        sci.space       |  194.4  |   91.0  |
  |      comp.graphics     |  157.8  |   62.0  |
  | soc.religion.christian |  201.9  |   83.0  |
  |       alt.atheism      |  202.3  |   82.0  |
  |   talk.politics.guns   |  262.8  |  157.0  |
  |    rec.sport.hockey    |  215.7  |  104.0  |

* **Уникальные слова после векторизации**:

  * Всего по корпусу: 39 254
  * По классам: от 8 737 (sci.space) до 14 367 (rec.sport.hockey)

Текст предобрабатывался через `CountVectorizer` с удалением стоп-слов, приведением к нижнему регистру и фильтрацией по частотам.

---

## 3. Результаты выделения тем

### 3.1. sklearn.decomposition.LatentDirichletAllocation

| Topic | Топ-слова                                                       |
| :---: | :-------------------------------------------------------------- |
|   0   | state crime pts law weapons control firearms file guns gun      |
|   1   | time christian faith does people christ bible church jesus god  |
|   2   | moon data lunar shuttle satellite orbit earth launch nasa space |
|   3   | jpeg files ftp data available software file graphics edu image  |
|   4   | good time does say know like just think people don              |
|   5   | league players year nhl games season play hockey game team      |

---

### 3.2. Собственная реализация LDA

| Topic | Топ-слова                                                             |
| :---: | :-------------------------------------------------------------------- |
|   0   | control year period team pts file firearms new guns gun               |
|   1   | time christian faith does people christ bible church jesus god        |
|   2   | moon data lunar shuttle satellite orbit earth launch nasa space       |
|   3   | jpeg files ftp data available software file graphics edu image        |
|   4   | good time does say know like does say know like just think don people |
|   5   | points van win goal hockey play games season team game                |

---

## 4. Сравнение качества и времени обучения

|    Реализация   | Coherence | Время `fit` (mean ± std)                       |
| :-------------: | :-------: | :--------------------------------------------- |
|   sklearn LDA   |   0.7767  | 9.05 s ± 4.53 s per loop (7 runs, 1 loop each) |
| Собственная LDA |   0.7147  | 27.7 s ± 7.4 s per loop (7 runs, 1 loop each)  |

---

**Выводы:**

* LDA позволяет извлекать читаемые темы, совпадающие между библиотечной и собственной реализацией.
* Оптимизированные C/Fortran-реализации (sklearn) выигрывают по времени и зачастую дают более стабильные оценки в силу более точного ELBO-контроля.
* Для дальнейшего ускорения собственной реализации стоит рассмотреть batch-обучение, JIT-компиляцию (Numba) или перевод «горячих» циклов на Cython.
