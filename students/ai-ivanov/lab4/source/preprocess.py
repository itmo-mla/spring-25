import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
try:
    word_tokenize("test")
except LookupError:
    nltk.download("punkt")
try:
    WordNetLemmatizer().lemmatize("tests")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.pos_tag(["test"])
except LookupError:
    nltk.download("averaged_perceptron_tagger")


class TextPreprocessor:
    """
    Класс для предобработки текстовых данных.

    Включает в себя:
    - Токенизацию
    - Приведение к нижнему регистру
    - Удаление пунктуации
    - Удаление стоп-слов
    - Лемматизацию
    """

    def __init__(self, language: str = "english"):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation_table = str.maketrans("", "", string.punctuation)

    def _get_wordnet_pos(self, word: str) -> str:
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": nltk.corpus.wordnet.ADJ,
            "N": nltk.corpus.wordnet.NOUN,
            "V": nltk.corpus.wordnet.VERB,
            "R": nltk.corpus.wordnet.ADV,
        }
        return tag_dict.get(
            tag, nltk.corpus.wordnet.NOUN
        )

    def preprocess_text(self, text: str) -> list[str]:
        """
        Предобрабатывает один текстовый документ.

        Parameters
        ----------
        text : str
            Входной текст для обработки.

        Returns
        -------
        list[str]
            Список обработанных токенов (слов).
        """
        if not isinstance(text, str):
            # logger.warning(f"Ожидалась строка, получено {type(text)}. Возвращен пустой список.")
            return []

        # 1. Токенизация
        tokens = word_tokenize(text.lower())  # Приведение к нижнему регистру сразу

        # 2. Удаление пунктуации
        tokens = [word.translate(self.punctuation_table) for word in tokens]

        # 3. Удаление пустых строк после удаления пунктуации и неалфавитных токенов
        tokens = [word for word in tokens if word.isalpha()]

        # 4. Удаление стоп-слов
        tokens = [word for word in tokens if word not in self.stop_words]

        # 5. Лемматизация с учетом части речи
        tokens = [
            self.lemmatizer.lemmatize(word, self._get_wordnet_pos(word))
            for word in tokens
        ]

        # Удаление очень коротких слов (например, длиной < 2) после лемматизации
        tokens = [word for word in tokens if len(word) > 1]

        return tokens

    def preprocess_documents(self, documents: list[str]) -> list[list[str]]:
        """
        Предобрабатывает список документов.

        Parameters
        ----------
        documents : list[str]
            Список текстовых документов.

        Returns
        -------
        list[list[str]]
            Список списков обработанных токенов.
        """
        processed_docs = []
        for doc in documents:
            processed_docs.append(self.preprocess_text(doc))
        return processed_docs


if __name__ == "__main__":
    sample_docs = [
        "This is the first document. It contains some punctuation and stop words!",
        "The second document is a bit longer and has more words for processing.",
        "And the third one? It's short & sweet.",
        "Numbers 123 and symbols like #$% should be handled.",
        "Running, ran, runs are all forms of run.",
        "",  # Пустой документ
        None,  # Некорректный ввод
    ]

    preprocessor = TextPreprocessor(language="english")

    print("Original Documents:")
    for i, doc in enumerate(sample_docs):
        print(f"Doc {i + 1}: {doc}")

    print("\nProcessed Documents:")
    processed_sample_docs = preprocessor.preprocess_documents(sample_docs)
    for i, doc_tokens in enumerate(processed_sample_docs):
        print(f"Doc {i + 1}: {doc_tokens}")

    # Пример одного документа
    sample_text = "This is a sample sentence with cats, dogs, and running exercises."
    processed_text = preprocessor.preprocess_text(sample_text)
    print(f"\nOriginal Text: {sample_text}")
    print(f"Processed Text: {processed_text}")

    # Пример с другим языком (если есть ресурсы NLTK)
    # try:
    #     preprocessor_russian = TextPreprocessor(language="russian")
    #     sample_russian_text = "Это пример предложения на русском языке с кошками, собаками и беговыми упражнениями."
    #     processed_russian_text = preprocessor_russian.preprocess_text(sample_russian_text)
    #     print(f"\nOriginal Russian Text: {sample_russian_text}")
    #     print(f"Processed Russian Text: {processed_russian_text}")
    # except Exception as e:
    #     print(f"\nCould not process Russian text (ensure NLTK resources for Russian are downloaded): {e}")
