from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle
import os
import psycopg2
from psycopg2 import sql

app = Flask(__name__)

# Database configuration
DB_NAME = 'automl_db'
DB_USER = 'automl_user'
DB_PASSWORD = 'yourpassword'
DB_HOST = 'localhost'
DB_PORT = '5432'

# Create a connection to the PostgreSQL database
def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

# Save dataset to PostgreSQL
def save_dataset_to_postgres(df, name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} ({});").format(
        sql.Identifier(name),
        sql.SQL(", ").join(
            sql.SQL("{} {}").format(
                sql.Identifier(col),
                sql.SQL('TEXT')
            ) for col in df.columns
        )
    ))
    for index, row in df.iterrows():
        cursor.execute(sql.SQL("INSERT INTO {} VALUES ({})").format(
            sql.Identifier(name),
            sql.SQL(', ').join([sql.Literal(row[col]) for col in df.columns])
        ))
    conn.commit()
    cursor.close()
    conn.close()

# Load dataset from PostgreSQL
def load_dataset_from_postgres(name):
    conn = get_db_connection()
    df = pd.read_sql_query(sql.SQL("SELECT * FROM {}").format(sql.Identifier(name)), conn)
    conn.close()
    return df

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    df = pd.read_csv(file)
    save_dataset_to_postgres(df, 'uploaded_data')
    return jsonify({"message": "File uploaded and data loaded successfully!"})

@app.route('/dataset-info', methods=['GET'])
def dataset_info():
    df = load_dataset_from_postgres('uploaded_data')
    if df is not None:
        info = {
            "columns": list(df.columns),
            "shape": df.shape
        }
        return jsonify(info)
    return jsonify({"message": "No data loaded!"})

@app.route('/remove-features', methods=['POST'])
def remove_features():
    df = load_dataset_from_postgres('uploaded_data')
    if df is not None:
        features = request.json['features']
        df.drop(columns=features, inplace=True)
        save_dataset_to_postgres(df, 'uploaded_data')
        return jsonify({"message": "Features removed successfully!"})
    return jsonify({"message": "No data loaded!"})

@app.route('/convert-to-numbers', methods=['POST'])
def convert_to_numbers():
    df = load_dataset_from_postgres('uploaded_data')
    if df is not None:
        df = pd.get_dummies(df)
        save_dataset_to_postgres(df, 'uploaded_data')
        return jsonify({"message": "Converted to numbers successfully!"})
    return jsonify({"message": "No data loaded!"})

@app.route('/train', methods=['POST'])
def train_models():
    df = load_dataset_from_postgres('uploaded_data')
    if df is not None:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        models = {
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier( ),
            "Ada Boost": AdaBoostClassifier(),
            "SVM": SVC(),
            "XGBoost": XGBClassifier(),
            "Bagging": BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10),
        }
        results = {}
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            model_path = os.path.join(MODEL_FOLDER, f'{model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            results[model_name] = {
                "accuracy": accuracy,
                "rmse": rmse
            }
        
        return jsonify(results)
    return jsonify({"message": "No data loaded!"})

MODEL_FOLDER = 'C:\Users\ visha\ automl_project\dataset\models'   

if __name__ == '__main__':
    app.run(debug=True)
