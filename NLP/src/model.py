from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class TextClassifier:
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        print(classification_report(y_test, predictions))
        print("Accuracy:", accuracy_score(y_test, predictions))
    
    def predict(self, new_data):
        # Predictslabels for new data
        predictions = self.model.predict(new_data)
        return predictions

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        print(classification_report(y_test, predictions, zero_division=0))
        print("Accuracy:", accuracy_score(y_test, predictions))
