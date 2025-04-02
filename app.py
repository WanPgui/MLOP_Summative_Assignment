import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

app = Flask(__name__)

# Paths
MODEL_PATH = 'models/model.pkl'
DATA_PATH = "diabetic_data.csv"
DB_PATH = 'diabetic_data.db'

# Create a database connection function for SQLite
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Load or train the model
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print('Model not found! Training a new model...')
        retrain_model()
    model = joblib.load(MODEL_PATH)
    print('Model loaded successfully.')

# Retrain the model with data from the database
def retrain_model():
    # Fetch data from SQLite database
    conn = get_db_connection()
    query = "SELECT * FROM diabetic_data"
    df = pd.read_sql(query, conn)

    # Preprocess data (age and weight handling)
    df['age'] = df['age'].map({
        '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
        '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
        '[80-90)': 8, '[90-100)': 9
    })
    df['weight'].replace('?', pd.NA, inplace=True)
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df['weight'].fillna(df['weight'].median(), inplace=True)
    df['diabetesMed'] = df['diabetesMed'].map({'Yes': 1, 'No': 0})

    features = ['age', 'time_in_hospital', 'num_medications', 'A1Cresult', 'max_glu_serum', 'diabetesMed']
    X = df[features]
    X = pd.get_dummies(X, drop_first=True)
    y = df['diabetesMed']

    # Train and save the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model retrained and saved.")

# Load and preprocess data from CSV
def load_and_preprocess_data(filepath=DATA_PATH):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset {filepath} not found.")
    data = pd.read_csv(filepath)
    data['diabetesMed'] = data['diabetesMed'].map({'Yes': 1, 'No': 0})
    X = data.drop(columns=['diabetesMed', 'encounter_id', 'patient_nbr'])
    X = pd.get_dummies(X, drop_first=True)
    y = data['diabetesMed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Preprocess input features for prediction
def preprocess_input(features):
    data = pd.read_csv(DATA_PATH)
    data['diabetesMed'] = data['diabetesMed'].map({'Yes': 1, 'No': 0})
    X = data.drop(columns=['diabetesMed', 'encounter_id', 'patient_nbr'])
    X = pd.get_dummies(X, drop_first=True)

    feature_columns = X.columns.tolist()
    feature_df = pd.DataFrame(features).T
    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df[feature_columns]
    return feature_df

# Visualizations
@app.route('/visualize', methods=['GET'])
def visualize():
    try:
        data = pd.read_csv(DATA_PATH)
        if 'diabetesMed' not in data.columns:
            return jsonify({'error': "'diabetesMed' column not found in the dataset!"}), 400
        sns.pairplot(data, hue="diabetesMed")
        plt.savefig("visualization.png")
        plt.close()
        return jsonify({"message": "Visualization saved!"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Model prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'features' in data:
            # Predict using the features from the input payload
            features = np.array(data['features']).reshape(1, -1)

            # Preprocess the features before prediction
            features = preprocess_input(features)
            prediction = model.predict(features)

        elif 'use_db' in data and data['use_db']:
            # Predict using data from the database
            conn = get_db_connection()
            query = "SELECT * FROM diabetic_data LIMIT 1;"  # Fetch one row from the database for testing
            df = pd.read_sql(query, conn)
            conn.close()

            # Preprocess the row (same as during training)
            df['age'] = df['age'].map({
                '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
                '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
                '[80-90)': 8, '[90-100)': 9
            })
            df['weight'].replace('?', pd.NA, inplace=True)
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df['weight'].fillna(df['weight'].median(), inplace=True)
            df['diabetesMed'] = df['diabetesMed'].map({'Yes': 1, 'No': 0})

            features = ['age', 'time_in_hospital', 'num_medications', 'A1Cresult', 'max_glu_serum', 'diabetesMed']
            X = df[features]
            X = pd.get_dummies(X, drop_first=True)

            # Preprocess the features before prediction
            features = preprocess_input(X.values)
            prediction = model.predict(features)

        else:
            return jsonify({'error': 'Invalid data provided. Please provide either "features" or "use_db" parameter.'}), 400

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Retraining the model endpoint
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        retrain_model()
        load_model()
        return jsonify({'message': 'Model retrained successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

## Load and preprocess data from CSV
def load_and_preprocess_data(filepath=DATA_PATH):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset {filepath} not found.")
    data = pd.read_csv(filepath)
    data['diabetesMed'] = data['diabetesMed'].map({'Yes': 1, 'No': 0})
    X = data.drop(columns=['diabetesMed', 'encounter_id', 'patient_nbr'])
    X = pd.get_dummies(X, drop_first=True)
    y = data['diabetesMed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Preprocess input features for prediction
def preprocess_input(features):
    data = pd.read_csv(DATA_PATH)
    data['diabetesMed'] = data['diabetesMed'].map({'Yes': 1, 'No': 0})
    X = data.drop(columns=['diabetesMed', 'encounter_id', 'patient_nbr'])
    X = pd.get_dummies(X, drop_first=True)

    feature_columns = X.columns.tolist()
    feature_df = pd.DataFrame(features).T
    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df[feature_columns]
    return feature_df

# Upload data and retrain model endpoint
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        
        # Check if the file has a valid extension and print its details
        if file and file.filename.endswith('.csv'):
            print(f"File received: {file.filename}")
            print(f"File content type: {file.content_type}")

            # Save the file to the uploads directory
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            # Try reading the file with pandas and catch any errors
            try:
                data = pd.read_csv(filepath, encoding='utf-8')
                print("File read successfully.")

                # Preprocess data
                data['age'] = data['age'].map({
                    '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
                    '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
                    '[80-90)': 8, '[90-100)': 9
                })
                data['weight'].replace('?', pd.NA, inplace=True)
                data['weight'] = pd.to_numeric(data['weight'], errors='coerce')
                data['weight'].fillna(data['weight'].median(), inplace=True)
                data['diabetesMed'] = data['diabetesMed'].map({'Yes': 1, 'No': 0})

                # Ensure the required columns are present, otherwise assign default values (e.g., NaN or 0)
                required_columns = ['num_medications', 'A1Cresult', 'max_glu_serum']
                for col in required_columns:
                    if col not in data.columns:
                        data[col] = pd.NA  # or you can fill with default values like 0 or other placeholders

                features = ['age', 'time_in_hospital', 'num_medications', 'A1Cresult', 'max_glu_serum', 'diabetesMed']
                X = data[features]
                X = pd.get_dummies(X, drop_first=True)  # Ensure categorical features are properly handled
                y = data['diabetesMed']

                # Train and save the model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)
                os.makedirs('models', exist_ok=True)
                joblib.dump(model, MODEL_PATH)

                return jsonify({"message": "Model retrained with uploaded data!"})

            except Exception as e:
                return jsonify({"error": f"Failed to read CSV file: {str(e)}"}), 400
        else:
            return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500
        

# Home route
@app.route('/')
def home():
    return jsonify({"message": "Flask app is running!"})

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port)  


