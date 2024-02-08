import pandas as pd
from preprocess import TextPreprocessor
from feature_extraction import FeatureExtractor
from model import TextClassifier
from sklearn.model_selection import train_test_split

def load_data(filename):
    data = pd.read_csv(filename)
    return data['Review'], data[' Sentiment']

def main():
    # Load and preprocess data
    reviews, sentiments = load_data('data/reviews.csv')
    preprocessed_reviews = [TextPreprocessor.preprocess(review) for review in reviews]
    
    features = FeatureExtractor.extract_features(preprocessed_reviews)
    
    X_train, X_test, y_train, y_test = train_test_split(features, sentiments, test_size=0.2, random_state=42)
    
    classifier = TextClassifier()
    classifier.train(X_train, y_train)
    classifier.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
