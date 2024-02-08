import pandas as pd
#   предназначен для предварительной обработки текстовых данных
from preprocess import TextPreprocessor
#   используется для извлечения признаков из предобработанных текстов
from feature_extraction import FeatureExtractor
#   представляет собой классификатор текста.
from model import TextClassifier
#   используется для разделения данных на обучающую и тестовую выборки.
from sklearn.model_selection import train_test_split


#   Читает csv файл и возвращает Метки признака
def load_data(filename):
    data = pd.read_csv(filename)
    return data['Review'], data[' Sentiment']

def main():
#   Загрузка и предварительная обработка данных
    reviews, sentiments = load_data('data/reviews.csv')
    preprocessed_reviews = [TextPreprocessor.preprocess(review) for review in reviews]
    
#   Извлекает признаки
    features = FeatureExtractor.extract_features(preprocessed_reviews)
    
#   Разделяет данные на обучающую и тестовую выборки
#   test_size=0.2 означает, что 20% данных для тестирования, а оставшиеся для обучения 
    X_train, X_test, y_train, y_test = train_test_split(features, sentiments, test_size=0.2, random_state=42)
    
#   создаем экземпляр
    classifier = TextClassifier()
#   обучает его выборке
    classifier.train(X_train, y_train)
#   оценивает проиизводительность 
    classifier.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
