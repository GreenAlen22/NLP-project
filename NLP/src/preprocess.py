import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')  # For tokenization

class TextPreprocessor:
    @staticmethod
    def preprocess(text):
        tokens = word_tokenize(text)
        
        tokens = [word for word in tokens if word not in string.punctuation]
        
        tokens = [word.lower() for word in tokens]
        
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
