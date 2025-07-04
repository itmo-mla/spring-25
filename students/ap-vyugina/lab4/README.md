# Лабораторная работа №4

## Данные
Для реализации и проверки методов выбран датасет `20 News Groups`. После загрузки он дополнительно обрабатывается: разбивается на слова, векторизатор также производит очистку, убирая слишком частые или слишком редкие слова. Затем происходит преобразование в разреженную термино-документную матрицу - по сути, числовое представление распределения слов по документам.

## Реализация метода
Латентное размещение Дирихле реализовано через EM-алгоритм. На E-шаге вычисляется распределение вероятностей принадлежности различным темам исходя из текущих оценок распределения, на M-шаге Обновляются оценки распределения тем в документах и распределение слов в темах, используя рассчитанные на предыдущем этапе ожидания. 

**Описание основных компонент модели:**
* `phi_wt`: Вероятность слова при заданной теме (`p(word | topic)`)
* `theta_td`: Вероятность темы в документе (`p(topic | document)`)
* `topic_assignments`: Матрица вероятностных назначений тем каждому слову документа (`p(topic | document, word)`)

Алгоритм останавливается либо после достижения максимального числа итераций, либо когда изменения в параметрах между итерациями становятся незначительными (менее порога 0.05).

### Расчет когерентности

Для расчёта когеренции моделей написана функция, способная принимать результаты моделей от разных фреймворков и уже при помощи библиотеки Gensim вычислять когерентность.

**Входные аргументы:**
* `topic_word_distribution`: Распределение частотности слов по темам, которое получается после тренировки модели LDA
* `feature_names`: Список всех уникальных слов (лексикон), используемых моделью. В данной ситуации получен от векторизатора.
* `texts`: тексты, на которых оценивается качество моделей
* `n_words`: число наиболее значимых слов, учитываемых при оценке когерентности каждой темы
* `n_topics`: число рассматриваемых тем

Для каждой темы выделяются её топовые слова (имеющие наибольший вес согласно матрице распределения слов по темам). Создается словарь (`gensim.corpora.dictionary.Dictionary`) из списка токенов текстов (необходим для построения меры когерентности).

Затем создаётся объект класса `CoherenceModel`, который принимает список выделенных слов каждой темы, сам корпус текстов и тип метрики ("c_v").
Возвращаемое значение — итоговая оценка когерентности тем, чем выше значение, тем лучше согласованность тем.

## Сравнение
Кастомный алгоритм сравнивался с двумя библиотечными реализациями: из библиотеки `Gensim` и `Sklearn`. Словари для них генерировались из одинакового набора слов, использовалось одни и те же параметры алгоритмов. 

Когерентность для каждого из алгоритмов и время на обучение представлены в таблице ниже.

| Модель  | Когерентность | Время работы, с |
|---------|-------|--------|
| Custom  | 0.372 | 212.27 |
| Gensim  | 0.357 | 8.7 |
| Sklearn | 0.388 | 2.247 |

Качество работы алгоритмов достаточно близко.

В целом низкая когерентность обусловлена тем, что из оригинального датасета выбираются только 1000 документов. Более того, эти документы выбираются рандомно, поэтому для каждой темы не очень много документов. В том случае, если бы были выбраны N тем и все документы для них, результат был бы лучше.