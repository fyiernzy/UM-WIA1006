# THIS IS A CODE TO TEST THE MODEL WHETHER IT IS WORKABLE IN IDE.
import joblib
import pandas as pd

class ModelTester:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def preprocess_input(self, input_data):
        input_data_modified = {
            'Age': int(input_data['age']),
            'RestingBP': int(input_data['blood_pressure']),
            'Cholesterol': int(input_data['cholesterol']),
            'FastingBS': int(input_data['fasting_bs']),
            'MaxHR': int(input_data['max_hr']),
            'Oldpeak': float(input_data['oldpeak']),
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

        return pd.DataFrame(input_data_modified, index=[0])

    def predict(self, input_data):

        preprocessed_data = self.preprocess_input(input_data)

        prediction = self.model.predict(preprocessed_data)
        return prediction[0]

model_path = 'best_knn_model.pkl' 
tester = ModelTester(model_path)

# Sample input data
input_data = {
    'age': '35',
    'blood_pressure': '120',
    'cholesterol': '200',
    'fasting_bs': '120',
    'max_hr': '180',
    'oldpeak': '2.5',
    'gender': 'M',
    'chest_pain_type': 'ASY',
    'resting_ecg': 'Normal',
    'exercise_angina': 'N',
    'st_slope': 'Up'
}

prediction = tester.predict(input_data)
print(f"Prediction: {prediction}")
