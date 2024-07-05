from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model using pickle
with open('spotify_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the expected feature names
with open('feature_names.txt', 'r') as f:
    expected_features = [line.strip() for line in f]

# Preprocess function to handle incoming data
def preprocess(data):
    df = pd.DataFrame(data)

    # Check if 'genre' column is present in the data
    if 'genre' not in df.columns:
        raise ValueError("Input data must contain 'genre' column.")

    # Impute missing values for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # One-hot encode the 'genre' column
    df = pd.get_dummies(df, columns=['genre'], drop_first=True)

    # Add missing expected features with default values
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0  # or an appropriate default value

    # Ensure the DataFrame contains only the expected features
    df = df[expected_features]

    return df

@app.route('/')
def home():
    return render_template('test_predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input data from the form
        input_data = request.form['input_data']
        data = json.loads(input_data)

        # Preprocess the data
        features = preprocess(data)

        # Make prediction
        prediction = model.predict(features)

        # Render the template with the prediction
        return render_template('test_predict.html', prediction_text=f'Prediction: {prediction[0]}')
    
    except Exception as e:
        # Handle exceptions and return a meaningful error message
        return render_template('test_predict.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
