# Imports.
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS

# Load the trained model.
model = joblib.load('best_knn_model.pkl')

# Create a Flask app.
app = Flask(__name__)
CORS(app)

# Create a route to handle the prediction request.
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.form.to_dict()
    
    # Modify the input data to include the necessary feature names
    input_data_modified = {
        'Age': input_data['age'],
        'RestingBP': input_data['blood_pressure'],
        'Cholesterol': input_data['cholesterol'],
        'FastingBS': input_data['fasting_bs'],
        'MaxHR': input_data['max_hr'],
        'Oldpeak': input_data['oldpeak'],
        'Sex_F': 1 if input_data['gender'] == 'F' else 0,
        'Sex_M': 1 if input_data['gender'] == 'M' else 0,
        'ChestPainType_ASY': 1 if input_data['chest_pain_type'] == 'ASY' else 0,
        'ChestPainType_ATA': 1 if input_data['chest_pain_type'] == 'ATA' else 0,
        'ChestPainType_NAP': 1 if input_data['chest_pain_type'] == 'NAP' else 0,
        'ChestPainType_TA': 1 if input_data['chest_pain_type'] == 'TA' else 0,
        'RestingECG_LVH': 1 if input_data['resting_ecg'] == 'LVH' else 0,
        'RestingECG_Normal': 1 if input_data['resting_ecg'] == 'Normal' else 0,
        'RestingECG_ST': 1 if input_data['resting_ecg'] == 'ST' else 0,
        'ExerciseAngina_N': 1 if input_data['exercise_angina'] == 'N' else 0,
        'ExerciseAngina_Y': 1 if input_data['exercise_angina'] == 'Y' else 0,
        'ST_Slope_Down': 1 if input_data['st_slope'] == 'Down' else 0,
        'ST_Slope_Flat': 1 if input_data['st_slope'] == 'Flat' else 0,
        'ST_Slope_Up': 1 if input_data['st_slope'] == 'Up' else 0
        }
        
    # Preprocess the input data in DataFrame.
    df = pd.DataFrame(input_data_modified, index=[0])

    # Make predictions using the loaded model.
    prediction = model.predict(df)
    prediction = prediction.tolist()
    
    # Return the prediction as JSON.
    result = {'prediction': prediction[0]}
    return jsonify(result)

#Create a route to render the HTML file.
@app.route('/')
def home():
    return render_template('frontend.html')

# Run the Flask app.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
