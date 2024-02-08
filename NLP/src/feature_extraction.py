from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    vectorizer = TfidfVectorizer()

    @staticmethod
    def extract_features(texts):
        features = FeatureExtractor.vectorizer.fit_transform(texts)
        return features

    @staticmethod
    def transform_new_text(new_texts):
        new_features = FeatureExtractor.vectorizer.transform(new_texts)
        return new_features
