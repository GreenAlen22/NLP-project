#   Класс для наивного байесовского классификатора,
from sklearn.naive_bayes import MultinomialNB
#   Функция для разделения набора данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
#   выводит различные метрики классификации, для каждого класса вычисляет общую точность модели
from sklearn.metrics import classification_report, accuracy_score

class TextClassifier:
#   создаем экземпляр для MultinomialNB Это наивный байесовский классификатор
#   который будет использоваться для обучения и предсказания
    def __init__(self):
        self.model = MultinomialNB()

#   принимает обучающие данные (X_train и y_train) и обучает на них 
#   модель с помощью метода fit
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    
#   используется для предсказания меток новых данных (new_data) 
#   с помощью обученной модели
    def predict(self, new_data):
        # Predictslabels for new data
        predictions = self.model.predict(new_data)
        return predictions

#   Метод принимает тестовые данные (X_test и y_test), делает предсказания 
#   с помощью модели, а затем выводит отчет о классификации и точность модели
    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        print(classification_report(y_test, predictions, zero_division=0))
        print("Точность:", accuracy_score(y_test, predictions))
