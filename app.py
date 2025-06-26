from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.compose._column_transformer import ColumnTransformer

app = Flask(__name__)

# Load model and preprocessor
# Monkey-patch for backward compatibility
if not hasattr(ColumnTransformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    ColumnTransformer._RemainderColsList = _RemainderColsList

# THEN load your preprocessor
preprocessor = joblib.load(r"C:\Users\HC\Downloads\heart disease prediction 1\preprocessor.pkl")
model = load_model(r"C:\Users\HC\Downloads\heart disease prediction 1\heart_disease_model.h5")  # Keras model
  # scikit-learn preprocessor

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = {
        "age": int(request.form['age']),
        "sex": int(request.form['sex']),
        "cp": int(request.form['cp']),
        "trestbps": float(request.form['trestbps']),
        "chol": float(request.form['chol']),
        "fbs": int(request.form['fbs']),
        "restecg": int(request.form['restecg']),
        "thalach": float(request.form['thalach']),
        "exang": int(request.form['exang']),
        "oldpeak": float(request.form['oldpeak']),
        "slope": int(request.form['slope']),
        "ca": int(request.form['ca']),
        "thal": int(request.form['thal'])
    }

    # Convert to DataFrame and preprocess
    input_df = pd.DataFrame([features])
    processed_input = preprocessor.transform(input_df)

    # Convert to dense array if necessary
    if not isinstance(processed_input, np.ndarray):
        processed_input = processed_input.toarray()

    # Predict with Keras model
    probability = model.predict(processed_input, verbose=0)[0][0]
    prediction = 1 if probability >= 0.5 else 0  # Threshold at 0.5

    return render_template('result.html',
                         prediction=prediction,
                         probability=f"{probability * 100:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)