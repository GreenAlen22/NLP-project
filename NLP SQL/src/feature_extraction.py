from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
#   создание атрибута в качестве экземпляра TfidfVectorizer
    vectorizer = TfidfVectorizer()

#   метод принимает на вход список текстов (texts) и использует vectorizer для 
#   вычисления признаков этих текстов. Метод fit_transform сначала обучается 
#   на переданных текстах (вычисляет IDF для каждого слова), а затем преобразует 
#   тексты в соответствующие векторы TF-IDF
    @staticmethod
    def extract_features(texts):
        features = FeatureExtractor.vectorizer.fit_transform(texts)
        return features

#   Этот метод такой же как extract_features, но нужен для преобразования новых 
#   текстовых данных, используя уже обученный vectorizer. Метод transform не обучается
#   на новых данных, он просто меняет их в ранее вычисленными IDF значениями
    def transform_new_text(new_texts):
        new_features = FeatureExtractor.vectorizer.transform(new_texts)
        return new_features
