# Латентное размещение Дирихле (Latent Dirichlet Allocation - LDA)

Латентное размещение Дирихле (LDA) — это порождающая вероятностная модель, широко используемая для **тематического моделирования**. Основная идея LDA заключается в том, что документы представляются как смесь различных тем, а каждая тема, в свою очередь, характеризуется определенным распределением слов. Это позволяет автоматически выявлять скрытые тематические структуры в больших коллекциях текстов.

## Основные идеи и допущения LDA

Модель LDA строится на нескольких ключевых допущениях:

1.  **Документы — это смесь тем:** Каждый документ рассматривается как набор тем с различными пропорциями. Например, новостная статья может на 70% относиться к теме "Политика", на 20% к теме "Экономика" и на 10% к теме "Международные отношения".
2.  **Темы — это распределения слов:** Каждая тема определяется набором слов и вероятностями их появления в данной теме. Например, тема "Технологии" будет с высокой вероятностью содержать слова "компьютер", "программа", "интернет", "инновации".
3.  **Модель "мешка слов" (Bag-of-Words):** Порядок слов в документе не учитывается, важно только их наличие и частота.
4.  **Фиксированное количество тем ($K$):** Аналитик заранее задает количество тем, которые модель должна обнаружить в коллекции документов.

## Как LDA генерирует документы (Порождающий процесс)

LDA предполагает следующий гипотетический процесс создания каждого документа в коллекции:

Пусть:
*   $M$ — общее количество документов.
*   $K$ — заданное количество тем.
*   $V$ — размер словаря (общее количество уникальных слов во всех документах).
*   $\alpha$ (альфа) — гиперпараметр, влияющий на распределение тем в документах. Обычно это скаляр.
*   $\beta$ (бета) — гиперпараметр, влияющий на распределение слов в темах. Обычно это скаляр.

Для каждого документа $d$ из $M$ документов:

1.  **Выбирается распределение тем для документа ($\theta_d$):**
    Для документа $d$ случайным образом выбирается $K$-мерный вектор $\theta_d = (\theta_{d,1}, \dots, \theta_{d,K})$, где $\theta_{d,k}$ — это вероятность темы $k$ в документе $d$ (причем $\sum_{k=1}^K \theta_{d,k} = 1$). Этот вектор выбирается из **распределения Дирихле** с параметром $\alpha$:
    $ \theta_d \sim \text{Dir}(\alpha) $
    *   Если $\alpha$ мало (например, < 1), документы обычно будут состоять из небольшого числа доминирующих тем.
    *   Если $\alpha$ велико (> 1), темы в документах будут распределены более равномерно.

Для каждой темы $k$ из $K$ тем:

2.  **Выбирается распределение слов для темы ($\phi_k$):**
    Для темы $k$ случайным образом выбирается $V$-мерный вектор $\phi_k = (\phi_{k,1}, \dots, \phi_{k,V})$, где $\phi_{k,v}$ — это вероятность слова $v$ в теме $k$ (причем $\sum_{v=1}^V \phi_{k,v} = 1$). Этот вектор также выбирается из **распределения Дирихле**, но с параметром $\beta$:
    $ \phi_k \sim \text{Dir}(\beta) $
    *   Если $\beta$ мало, темы будут характеризоваться небольшим набором специфичных, часто встречающихся в них слов.
    *   Если $\beta$ велико, слова в темах будут распределены более равномерно.

Для каждого слова $w_{d,n}$ в документе $d$ (где $n$ — позиция слова в документе):

3.  **а. Выбирается тема для слова ($z_{d,n}$):**
    Из $K$ тем выбирается одна тема $z_{d,n}$ в соответствии с распределением тем для данного документа $\theta_d$:
        $ z_{d,n} \sim \text{Multinomial}(\theta_d) $

4.  **б. Выбирается само слово ($w_{d,n}$):**
    Из словаря выбирается слово $w_{d,n}$ в соответствии с распределением слов для выбранной на предыдущем шаге темы $\phi_{z_{d,n}}$:
        $ w_{d,n} \sim \text{Multinomial}(\phi_{z_{d,n}}) $

Этот процесс повторяется для каждого слова в каждом документе. Распределение Дирихле здесь используется потому, что оно хорошо подходит для моделирования распределений вероятностей (ведь $\theta_d$ и $\phi_k$ — это векторы вероятностей, компоненты которых суммируются в 1).

## Обучение модели LDA (Вывод скрытых переменных)

Задача обучения LDA — по коллекции наблюдаемых документов $W$ (набору слов) и заданным гиперпараметрам $\alpha$ и $\beta$ оценить скрытые (латентные) переменные:
*   $\Theta = \{\theta_d\}$: распределения тем по документам.
*   $\Phi = \{\phi_k\}$: распределения слов по темам.
*   $Z = \{z_{d,n}\}$: присвоение тем каждому слову.

Точное вычисление этих переменных аналитически невозможно из-за сложности модели. Поэтому используются приближенные методы, наиболее популярным из которых является **сэмплирование Гиббса**.

### Алгоритм `fit`: Обучение модели с помощью сэмплирования Гиббса

Метод `fit` обучает модель LDA на предоставленном корпусе документов. Вот его основные шаги:

1.  **Подготовка данных и инициализация:**
    *   **Построение словаря:** Из всех документов корпуса собираются уникальные слова, формируя словарь. Каждому слову присваивается уникальный ID. Документы преобразуются в последовательности ID слов.
    *   **Инициализация счетчиков:** Создаются матрицы для подсчета:
        *   `doc_topic_counts[d, k]` ($N_{d,k}$): сколько раз слова из документа $d$ были отнесены к теме $k$.
        *   `topic_word_counts[k, w]` ($N_{k,w}$): сколько раз слово $w$ (его ID) было отнесено к теме $k$ по всему корпусу.
        *   `doc_lengths[d]` ($N_d$): общее количество слов в документе $d$.
        *   `topic_counts[k]` ($N_k$): общее количество слов, отнесенных к теме $k$ по всему корпусу.
    *   **Случайное присвоение тем:** Каждому слову в каждом документе случайным образом присваивается одна из $K$ тем. После этого все счетчики обновляются в соответствии с этими первоначальными присвоениями.

2.  **Итеративное сэмплирование Гиббса:**
    Этот процесс повторяется заданное количество раз (`n_iter`). В каждой итерации для каждого слова $w_i$ (с ID `word_id`) в каждом документе $d$:
    a.  **"Удаление" слова из модели:** Текущее присвоение темы слову $w_i$ отменяется. Соответствующие счетчики $N_{d, s}$, $N_{s, \text{word\_id}}$, $N_d$ (не меняется), $N_s$ уменьшаются на единицу. Эти уменьшенные счетчики будем обозначать как $N^{
        eg i}_{d,k}$, $N^{
        eg i}_{k,w}$, $N^{
        eg i}_{k}$ (где $N_d$ не меняется, но для знаменателя используется $N_d-1$, т.е. длина документа без текущего слова).

    b.  **Вычисление вероятностей для каждой темы:** Для каждой темы $k \in \{1, \dots, K\}$ вычисляется условная вероятность того, что слово $w_i$ принадлежит теме $k$. Эта вероятность пропорциональна произведению двух факторов:
        *   **Вероятность темы $k$ в документе $d$:** Насколько тема $k$ "популярна" в текущем документе $d$ (без учета текущего слова $w_i$).
            $ P(k | d) \propto N^{
                eg i}_{d,k} + \alpha $
        *   **Вероятность слова $w_i$ в теме $k$:** Насколько слово $w_i$ "характерно" для темы $k$ (без учета текущего экземпляра слова $w_i$).
            $ P(w_i | k) \propto N^{
                eg i}_{k, w_i} + \beta $
        Полная формула для вероятности присвоения слова $w_i$ (с ID `word_id`) теме $k$ (с учетом нормализации и гиперпараметров $\alpha, \beta$, количества тем $K$ и размера словаря $V$):
        $$
        p(z_i = k | \mathbf{z}_{
            eg i}, \mathbf{w}, \alpha, \beta) \propto \frac{N^{
            eg i}_{d,k} + \alpha}{ (N_d - 1) + K\alpha } \times \frac{N^{
            eg i}_{k, \text{word\_id}} + \beta}{ (N^{
            eg i}_k) + V\beta }
        $$
        Где:
        *   $N^{
            eg i}_{d,k}$: число слов из документа $d$, отнесенных к теме $k$ (не считая текущее слово $w_i$).
        *   $N_d - 1$: общее число слов в документе $d$ (не считая текущее слово $w_i$).
        *   $N^{
            eg i}_{k, \text{word\_id}}$: число раз, когда слово $w_i$ было отнесено к теме $k$ по всему корпусу (не считая текущий экземпляр).
        *   $N^{
            eg i}_k$: общее число слов, отнесенных к теме $k$ по всему корпусу (не считая текущее слово $w_i$).
        *   $K\alpha$ и $V\beta$: сглаживающие добавки, использующие гиперпараметры.

    c.  **Выбор новой темы:** Новая тема для слова $w_i$ выбирается (сэмплируется) из $K$ тем в соответствии с распределением вероятностей, вычисленным на шаге (b).

    d.  **Обновление счетчиков:** Слову $w_i$ присваивается новая выбранная тема. Счетчики $N_{d, n}$, $N_{n, \text{word\_id}}$, $N_d$ (не меняется), $N_n$ увеличиваются на единицу.

3.  **Расчет финальных распределений:**
    После завершения всех итераций сэмплирования, финальные распределения вычисляются на основе накопленных счетчиков:
    *   **Распределение слов по темам ($\Phi$):** Вероятность слова $w$ (ID `word_id`) в теме $k$.
        $ \phi_{k,w} = P(w|k) = \frac{N_{k, \text{word\_id}} + \beta}{N_k + V \cdot \beta} $
    *   **Распределение тем по документам ($\Theta$):** Вероятность темы $k$ в документе $d$.
        $ \theta_{d,k} = P(k|d) = \frac{N_{d,k} + \alpha}{N_d + K \cdot \alpha} $

Эти матрицы $\Phi$ (или `topic_word_distribution_`) и $\Theta$ (или `doc_topic_distribution_`) и являются результатом работы обученной модели LDA.

### Алгоритм `transform`: Применение обученной модели к новым документам

Метод `transform` позволяет получить тематические распределения для новых документов, которые не использовались при обучении модели. Он использует уже вычисленные на этапе `fit` распределения слов по темам ($\Phi$).

1.  **Преобразование новых документов:** Слова в новых документах переводятся в ID из словаря, созданного на этапе обучения. Неизвестные слова (которых нет в словаре) игнорируются.
2.  **Инициализация для новых документов:** Для каждого нового документа $d_{new}$ инициализируются счетчики `doc_topic_counts_new[d_new, k]` (сколько слов из $d_{new}$ отнесено к теме $k$). Словам случайным образом присваиваются темы, и счетчики обновляются.
3.  **Сокращенные итерации сэмплирования:** Выполняется небольшое количество итераций сэмплирования Гиббса (обычно меньше, чем при обучении). Для каждого слова $w$ в новом документе $d_{new}$:
        a.  Уменьшается счетчик `doc_topic_counts_new` для текущей темы слова.
    b.  Вычисляется условная вероятность присвоения слова $w$ каждой теме $k$. Здесь используется та же логика, что и в `fit`, но вероятность слова в теме $P(w|k)$ берется из уже обученной матрицы $\Phi$:
        $ p(z_i = k | \dots) \propto P(k|d_{new}) \times \phi_{k,w} $
        Где $P(k|d_{new}) = \frac{doc\_topic\_counts\_new_{d_{new},k} + \alpha}{\text{длина}(d_{new}) - 1 + K \cdot \alpha}$.
    c.  Новая тема для слова $w$ сэмплируется.
        d.  Счетчик `doc_topic_counts_new` обновляется.
4.  **Расчет финальных распределений тем:** После итераций вычисляется распределение тем для каждого нового документа ($\Theta_{new}$):
    $ \theta_{d_{new},k} = P(k|d_{new}) = \frac{doc\_topic\_counts\_new_{d_{new},k} + \alpha}{\text{длина}(d_{new}) + K \cdot \alpha} $
    (где `длина(d_new)` — количество известных слов в документе $d_{new}$).

## Применение LDA

LDA находит применение в различных задачах анализа текстов:
*   **Тематический анализ коллекций документов:** Выявление доминирующих тем.
*   **Классификация и кластеризация текстов:** Использование тем как признаков.
*   **Улучшение информационного поиска:** Сопоставление запросов с темами документов.
*   **Рекомендательные системы:** Предложение контента на основе тематической схожести.
*   **Анализ трендов и мнений** в социальных медиа, новостях и научных публикациях.

## Важные аспекты: Предобработка и Оценка

*   **Предобработка текста:** Качество тем, обнаруживаемых LDA, сильно зависит от подготовки исходных текстов. Стандартные шаги включают:
    *   **Токенизация:** Разделение текста на отдельные слова (токены).
    *   **Удаление стоп-слов:** Исключение часто встречающихся, но малоинформативных слов (например, "и", "в", "на", "быть").
    *   **Лемматизация или стемминг:** Приведение слов к их начальной форме (например, "иду", "шёл", "идёт" -> "идти").
    *   **Фильтрация:** Удаление слишком редких или слишком частых слов, не несущих специфической тематической нагрузки.
*   **Оценка качества модели:** Это непростая задача. Используются:
    *   **Перплексия (Perplexity):** Метрика, показывающая, насколько хорошо модель предсказывает новые (невиданные) данные. Обычно чем ниже перплексия, тем лучше, но это не всегда коррелирует с человеческой интерпретируемостью тем.
    *   **Когерентность тем (Topic Coherence):** Оценивает, насколько слова внутри одной темы семантически связаны и понятны человеку. Существуют различные метрики когерентности (например, C_v, UMass).
    *   **Качественная оценка экспертами:** Анализ осмысленности и полезности выделенных тем специалистами в предметной области.

LDA — это мощный инструмент, но для получения качественных результатов он требует внимательной настройки гиперпараметров ($K, \alpha, \beta$), качественной предобработки данных и тщательной интерпретации полученных тем.

## Обучние модели

### Набор данных

https://www.kaggle.com/datasets/crawford/20-newsgroups/data

Context

This dataset is a collection newsgroup documents. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering.

Content

There is file (list.csv) that contains a reference to the document_id number and the newsgroup it is associated with.
There are also 20 files that contain all of the documents, one document per newsgroup.


### Результаты обучения

Время обучения: 18.5 секунды

#### Своя реализация

| Тема   | Топ-5 слов                               |
| :----- | :--------------------------------------- |
| Тема 1 | `space`, `technology`, `research`, `society`, `issue` |
| Тема 2 | `space`, `mission`, `orbit`, `probe`, `launch` |
| Тема 3 | `widget`, `use`, `resource`, `application`, `value` |
| Тема 4 | `god`, `atheist`, `nt`, `religion`, `believe` |
| Тема 5 | `period`, `pp`, `power`, `play`, `scorer` |
| Тема 6 | `drive`, `disk`, `system`, `hard`, `controller` |
| Тема 7 | `rate`, `gun`, `homicide`, `handgun`, `vancouver` |
| Тема 8 | `thanks`, `email`, `mouse`, `offer`, `call` |
| Тема 9 | `use`, `driver`, `window`, `program`, `file` |
| Тема 10 | `tax`, `court`, `mr`, `case`, `income` |
| Тема 11 | `space`, `nasa`, `available`, `information`, `data` |
| Тема 12 | `god`, `sin`, `say`, `christ`, `shall` |
| Тема 13 | `entry`, `file`, `output`, `program`, `section` |
| Тема 14 | `writes`, `article`, `kill`, `mother`, `henry` |
| Тема 15 | `db`, `mov`, `bh`, `byte`, `si` |
| Тема 16 | `game`, `team`, `win`, `run`, `player` |
| Тема 17 | `nt`, `one`, `would`, `make`, `get` |
| Тема 18 | `key`, `ripem`, `use`, `rsa`, `pgp` |
| Тема 19 | `armenian`, `russian`, `people`, `turk`, `muslim` |
| Тема 20 | `dod`, `ride`, `denizen`, `motorcycle`, `flame` |

#### Scikit-learn реализация

Время обучения: 2.5 секунды

| Тема   | Топ-5 слов (scikit-learn)              |
| :----- | :--------------------------------------- |
| Тема 1 | `say`, `people`, `prophecy`, `armenian`, `dead` |
| Тема 2 | `writes`, `article`, `nt`, `right`, `titan` |
| Тема 3 | `mr`, `say`, `book`, `case`, `writes` |
| Тема 4 | `armenian`, `russian`, `turk`, `turkish`, `army` |
| Тема 5 | `db`, `probe`, `space`, `bh`, `mission` |
| Тема 6 | `key`, `ripem`, `use`, `period`, `rsa` |
| Тема 7 | `widget`, `use`, `application`, `resource`, `value` |
| Тема 8 | `entry`, `file`, `output`, `program`, `section` |
| Тема 9 | `god`, `nt`, `atheist`, `say`, `religion` |
| Тема 10 | `drive`, `disk`, `hard`, `controller`, `bios` |
| Тема 11 | `space`, `nasa`, `shuttle`, `mission`, `data` |
| Тема 12 | `run`, `game`, `hit`, `nt`, `writes` |
| Тема 13 | `space`, `use`, `driver`, `box`, `file` |
| Тема 14 | `nt`, `thanks`, `like`, `use`, `problem` |
| Тема 15 | `convenient`, `value`, `book`, `kmail`, `parallel` |
| Тема 16 | `writes`, `article`, `know`, `nt`, `like` |
| Тема 17 | `dod`, `motorcycle`, `ride`, `nt`, `know` |
| Тема 18 | `nt`, `people`, `make`, `right`, `writes` |
| Тема 19 | `game`, `article`, `team`, `run`, `writes` |
| Тема 20 | `gun`, `rate`, `homicide`, `handgun`, `vancouver` |

## Сравнение качества моделей

### Перплексия (Perplexity)

Перплексия - это метрика, которая показывает, насколько хорошо модель предсказывает новые данные. Чем ниже значение перплексии, тем лучше модель. Формально, перплексия - это экспонента от средней отрицательной логарифмической правдоподобности на одно слово. Она показывает, насколько "удивлена" модель при встрече с новыми данными.

| Реализация | Перплексия |
|------------|------------|
| Custom LDA | 1426.0     |
| Scikit-learn LDA | 1726.5928 |

Вывод: Наша собственная реализация LDA показывает лучшие результаты с точки зрения перплексии (1426.0 против 1726.5928 у scikit-learn).
