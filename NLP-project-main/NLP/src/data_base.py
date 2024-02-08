#   Эта библиотека позволяет Python скриптам подключаться к базе 
#   данных PostgreSQL и выполнять SQL запросы
import psycopg2




#   Параметры подключения к базе данных
db_params = {
        'database': 'NLP_DB_PROJECT',
        'user': 'postgres',
        'password': '47998574',
        'host': 'localhost',
        'port': 5432
    }



def load_data_from_db(db_params):
#   Загружает данные из базы данных PostgreSQL
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

#   запрос для выборки данных из таблицы reviews.
    cursor.execute("SELECT review, sentiment FROM reviews;")

#   сохраняет в переменной data. fetchall() все строки запроса 
    data = cursor.fetchall()
    cursor.close()
    conn.close()

#   создается список с данными
    reviews = [row[0] for row in data]
    sentiments = [row[1] for row in data]
    return reviews, sentiments




def add_column_if_not_exists(db_params):
#   Загружает данные из базы данных PostgreSQL
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    
#    Проверяем, существует ли колонка в таблице
    cursor.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name='reviews' AND column_name='predicted_sentiment';
    """)
    
    result = cursor.fetchone()
    if not result:
#       Если колонка не существует, добавляем её
        cursor.execute("""
        ALTER TABLE reviews
        ADD COLUMN predicted_sentiment VARCHAR(255);
        """)
        print("Column 'predicted_sentiment' added to 'reviews'.")
    else:
        print("Column 'predicted_sentiment' already exists.")
    
    conn.commit()
    cursor.close()
    conn.close()

#   создаем таблице с результатами после окончания NLP
def create_results_table(db_params):
#   Подключаемся к базе данных
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    
#   Создаем SQL-запрос
    create_table_query = """
    CREATE TABLE IF NOT EXISTS results (
        review_id SERIAL PRIMARY KEY,
        predicted_sentiment VARCHAR(255)
    );
    """
    
#   Выполняем запрос
    cursor.execute(create_table_query)
    conn.commit()
    
#   Закрываем соединение
    cursor.close()
    conn.close()
    
    print("Table 'results' created successfully.")

#сохраннение результатов текста
def save_results_to_db(db_params, review_ids, predictions):
#   Подключаемся к базе данных
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

#   Проходится по парам идентификаторов отзывов и предсказанных настроений
#   zip() используется для чтению двух списков параллельно
#   SQL запрос для вставки данных в таблицу results
    for review_id, sentiment in zip(review_ids, predictions):
        cursor.execute("""
            INSERT INTO results (review_id, predicted_sentiment)
            VALUES (%s, %s)
            ON CONFLICT (review_id)
            DO UPDATE SET predicted_sentiment = EXCLUDED.predicted_sentiment;
        """, (review_id, sentiment))
        
    conn.commit()
    cursor.close()
    conn.close()



