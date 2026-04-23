# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))

# Feature names from your dataset
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

target_names = ['Malignant', 'Benign']

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    global df
    
    try:
        # Get all form values
        input_features = [float(x) for x in request.form.values()]
        
        # Validate we have the right number of features
        if len(input_features) != len(feature_names):
            return render_template('index.html', 
                                 prediction_text='Error: Please provide all 30 feature values',
                                 features=feature_names,
                                 show_result=True,
                                 result_class='error-result')
        
        # Convert to numpy array and reshape for prediction
        features_value = np.array(input_features).reshape(1, -1)
        
        # Make prediction
        prediction = breast_cancer_detector_model.predict(features_value)[0]
        
        # Get prediction probabilities if available
        try:
            prediction_proba = breast_cancer_detector_model.predict_proba(features_value)[0]
            confidence = prediction_proba[prediction] * 100
        except:
            confidence = "N/A"
        
        # Get prediction label
        result = target_names[prediction]
        
        # Store input and prediction in dataframe
        input_dict = {feature: value for feature, value in zip(feature_names, input_features)}
        input_dict['Prediction'] = result
        input_dict['Confidence'] = f"{confidence:.2f}%" if confidence != "N/A" else "N/A"
        
        df = pd.concat([df, pd.DataFrame([input_dict])], ignore_index=True)
        df.to_csv('breast_cancer_predictions.csv', index=False)
        
        # Determine message and styling based on prediction
        if prediction == 1:  # Benign
            message = "Great news! The tumor appears to be BENIGN (non-cancerous)."
            result_class = "benign-result"
            icon = "✅"
        else:  # Malignant
            message = "The tumor appears to be MALIGNANT (cancerous). Please consult with a healthcare professional for further evaluation."
            result_class = "malignant-result"
            icon = "⚠️"
            
        return render_template('index.html', 
                             prediction_text=f'{icon} Prediction: {result}',
                             confidence_text=f'Confidence: {confidence:.2f}%' if confidence != "N/A" else 'Confidence: N/A',
                             message=message,
                             features=feature_names,
                             show_result=True,
                             result_class=result_class)
                             
    except ValueError:
        return render_template('index.html', 
                             prediction_text='❌ Error: Please enter valid numerical values for all features',
                             features=feature_names,
                             show_result=True,
                             result_class='error-result')
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'❌ Error: {str(e)}',
                             features=feature_names,
                             show_result=True,
                             result_class='error-result')

@app.route('/view_data')
def view_data():
    global df
    if df.empty:
        return "No predictions made yet."
    return render_template('data.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

@app.route('/sample_data')
def sample_data():
    """Provide sample feature values for testing"""
    # Sample values (you can replace these with actual mean values from your dataset)
    sample_features = {
        'mean radius': 12.15,
        'mean texture': 17.85,
        'mean perimeter': 78.75,
        'mean area': 462.79,
        'mean smoothness': 0.089,
        'mean compactness': 0.081,
        'mean concavity': 0.049,
        'mean concave points': 0.021,
        'mean symmetry': 0.181,
        'mean fractal dimension': 0.062,
        'radius error': 0.271,
        'texture error': 1.250,
        'perimeter error': 1.958,
        'area error': 23.56,
        'smoothness error': 0.006,
        'compactness error': 0.021,
        'concavity error': 0.026,
        'concave points error': 0.011,
        'symmetry error': 0.020,
        'fractal dimension error': 0.003,
        'worst radius': 13.37,
        'worst texture': 23.39,
        'worst perimeter': 88.05,
        'worst area': 551.68,
        'worst smoothness': 0.123,
        'worst compactness': 0.203,
        'worst concavity': 0.165,
        'worst concave points': 0.077,
        'worst symmetry': 0.290,
        'worst fractal dimension': 0.079
    }
    return render_template('index.html', sample_features=sample_features, features=feature_names)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
