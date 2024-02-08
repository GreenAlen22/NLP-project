import pandas as pd
#   предназначен для предварительной обработки текстовых данных
from preprocess import TextPreprocessor
#   используется для извлечения признаков из предобработанных текстов
from feature_extraction import FeatureExtractor
#   представляет собой классификатор текста.
from model import TextClassifier
#   используется для разделения данных на обучающую и тестовую выборки.
from sklearn.model_selection import train_test_split

#   читает функции из data_base 
from data_base import load_data_from_db
from data_base import db_params
from data_base import save_results_to_db
from data_base import add_column_if_not_exists
from data_base import create_results_table


from sklearn.metrics import classification_report, accuracy_score


def main():
    # Загрузка и предварительная обработка данных
    reviews, sentiments = load_data_from_db(db_params)
    preprocessed_reviews = [TextPreprocessor.preprocess(review) for review in reviews]
    
    # Извлекает признаки
    features = FeatureExtractor.extract_features(preprocessed_reviews)
    
    # Разделяет данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(features, sentiments, test_size=0.2, random_state=42)
    
    # создаем экземпляр
    classifier = TextClassifier()
    # обучает его выборке
    classifier.train(X_train, y_train)
    # Получаем предсказания
    predictions = classifier.predict(X_test)
    # Сохраняем результаты предсказаний в базу данных
    # Предполагается, что у нас есть идентификаторы для X_test
    review_ids = range(X_test.shape[0])  # Это пример. Необходимо получить реальные идентификаторы.
    save_results_to_db(db_params, review_ids, predictions)


    # оценивает проиизводительность 
    print(classification_report(y_test, predictions, zero_division=0))
    print("Точность:", accuracy_score(y_test, predictions))
    add_column_if_not_exists(db_params)
    create_results_table(db_params)


if __name__ == "__main__":
    main()
