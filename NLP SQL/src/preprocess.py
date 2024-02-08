#Основной модуль для работы с NLP в Python
import nltk
# Модуль для работы со стоп-словами (словами, которые обычно исключаются из обработки из-за их частой встречаемости и бесполезности)
from nltk.corpus import stopwords
#Функция для разбиения текста на токены
from nltk.tokenize import word_tokenize
#   Класс для приведения слов к их базовой форме (в словарном ввиде например глаголу станут домов, дому → дом)
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords') #    для загрузки списка стоп-слов.
nltk.download('wordnet') #  для загрузки WordNet, используемого лемматизатором
nltk.download('punkt')  # для токенизации+

class TextPreprocessor:
    @staticmethod
#   принемает текст
    def preprocess(text):
#        разбиение текста на отдельные слова.
        tokens = word_tokenize(text)
#       Удаляет знаки пунктуации (точка запятая)
        tokens = [word for word in tokens if word not in string.punctuation]
#       преобразование всех слов в нижний регистр для унификации.
        tokens = [word.lower() for word in tokens]
        
#       Удаление стоп-слов
        stop_words = set(stopwords.words('english')) | set(stopwords.words('russian'))
        tokens = [word for word in tokens if word not in stop_words]
        
#       приведение слов к их базовой форме
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
#       возвращает текст в виде чистых слов и разедленные пробелом
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
