# Flask App with Model Deployment

This repository contains the code for a Flask web application that loads a machine learning model, retrains it if necessary, and exposes it via RESTful APIs. The app has been deployed using Docker and hosted on Render. It was initially developed and tested in Google Colab, with model training, endpoints testing, and deployment handled using several tools.

## Table of Contents

1. [Google Colab Model Training](#google-colab-model-training)
2. [Flask App Setup and Ngrok](#flask-app-setup-and-ngrok)
3. [Testing Endpoints](#testing-endpoints)
4. [Dockerizing the Flask App](#dockerizing-the-flask-app)
5. [Deploying on Render](#deploying-on-render)
6. [Project Structure](#project-structure)

---

link to the demo video: https://drive.google.com/file/d/1RXgik6iUGlN7PEeijXxc-rF_4onZeyU8/view?usp=sharing

link to the youtube link: https://youtu.be/Fjij74rheRU
## Google Colab Model Training

### 1. **Dataset and Model Setup**

- Initially, the machine learning model was trained in Google Colab using a dataset (`diabetic_data.csv`) to predict diabetes-related outcomes.  
- Data was cleaned, preprocessed, and split into features and target variables.

### 2. **Model Training**

```python
# Sample code from Colab to load and train the model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('diabetic_data.csv')

# Preprocessing
X = df.drop(columns=['diabetesMed'])  # Example target variable
y = df['diabetesMed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
import joblib
joblib.dump(model, 'diabetes_model.pkl')
```


### 3. **Flask App Setup and Ngrok**
To expose the Flask app over the internet, Ngrok was used.

Start Flask app:
python app.py

Start Ngrok:

In another terminal, run:
ngrok http 5000

This will provide you with a public URL to test the API externally.

###. 4 **Testing Endpoints**
1. Flask API for Prediction
To create a REST API that serves the model, Flask was used. Below is a sample Flask route for making predictions:

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('diabetes_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({"prediction": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

2. Testing the API Locally
To test the /predict endpoint locally, you can use tools like Postman or curl:

curl --location --request POST 'https://2201-34-106-144-131.ngrok-free.app/retrain' for retraining the model

curl -X GET https://9ec5-34-106-234-182.ngrok-free.app/visualize For visualizations images

curl -X POST `https://9ec5-34-106-234-182.ngrok-free.app/upload -H "Content-Type: Multipart/Form-Data" for uploading new data and retraining 

curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [1, 2, 3, 4, 5]}' for diabetic state prediction
curl --location 'https://9926-34-106-234-182.ngrok-free.app/predict' \
--header 'Content-Type: application/json' \
--data '{
  "features": [43, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}
'



### .5 **Dockerizing the Flask App**
1. Create Dockerfile
The Flask app was Dockerized using the following Dockerfile:

dockerfile
Copy
Edit
# Use the official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy necessary files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]

2. Build and Run Docker Image
Run the following commands to build and run the Docker image:

# Build the Docker image
docker build -t my-flask-app .

# Run the Docker container
docker run -it -p 5000:5000 my-flask-app
This will allow you to access the Flask API on http://localhost:5000.

### .6 **Deploying on Render**
1. Prepare for Deployment
Ensure that all necessary files (app.py, Dockerfile, requirements.txt, diabetic_data.csv, and model files) are present in the repository.

Create a Render account and link the GitHub repository with your project.

2. Create a New Web Service on Render
Log in to Render and create a new Web Service.

Connect your GitHub repository to Render.

Choose Docker as the environment, and specify the Dockerfile path if needed.

Set up environment variables if required (e.g., database credentials, API keys).

https://mlop-summative-assignment.onrender.com 

api access to the render web service connect:
35.160.120.126
44.233.151.27
34.211.200.85

https://dashboard.render.com/web/srv-cvmdc69r0fns738u85b0/deploys/dep-cvmk96ruibrs73bikdhg?r=2025-04-02%4014%3A06%3A24%7E2025-04-02%4014%3A11%3A18


3. Deploy the Application
Once the Web Service is created, Render will automatically build the Docker container and deploy the app. You can access the deployed application via the provided URL.

### .7 **Project Structure**

my-flask-app/
│
├── app.py               # Flask app for serving predictions
├── Dockerfile           # Docker configuration for building the container
├── requirements.txt     # Python dependencies
├── diabetic_data.csv    # Dataset
├── diabetes_model.pkl   # Trained model file
└── README.md            # This file
Conclusion
This project demonstrates the process of training a machine learning model, exposing it via a Flask API, testing it locally, Dockerizing the application, and finally deploying it to Render. The model is trained and retrained as needed, and the predictions can be made via a simple HTTP API.
