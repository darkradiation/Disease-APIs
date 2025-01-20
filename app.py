from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load all the pre-trained models
diabetes_model = joblib.load("models/diabetes_model.sav")
# diabetes_model = joblib.load("models/diabetes_prediction_decision_tree.sav")
heart_model = joblib.load("models/heart_disease_model.sav")
parkinson_model = joblib.load("models/parkinsons_model.sav")
lung_cancer_model = joblib.load("models/lung_cancer_model.sav")
breast_cancer_model = joblib.load("models/breast_cancer.sav")
chronic_disease_model = joblib.load("models/chronic_model.sav")
hepatitis_model = joblib.load("models/hepititisc_model.sav")
liver_model = joblib.load("models/liver_model.sav")

@app.route('/api/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.json
        required_fields = ['pregnancies', 'glucose', 'bloodPressure', 'skinThickness', 'insulin', 'bmi', 'diabetesPedigreeFunction', 'age']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = [
            data['pregnancies'],
            data['glucose'],
            data['bloodPressure'],
            data['skinThickness'],
            data['insulin'],
            data['bmi'],
            data['diabetesPedigreeFunction'],
            data['age']
        ]

        prediction = diabetes_model.predict([input_data])
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/heart', methods=['POST'])
def predict_heart():
    try:
        data = request.json
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = [
            data['age'],
            data['sex'],
            data['cp'],
            data['trestbps'],
            data['chol'],
            data['fbs'],
            data['restecg'],
            data['thalach'],
            data['exang'],
            data['oldpeak'],
            data['slope'],
            data['ca'],
            data['thal']
        ]

        prediction = heart_model.predict([input_data])
        result = "Heart Disease" if prediction[0] == 1 else "No Heart Disease"
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parkinson', methods=['POST'])
def predict_parkinson():
    try:
        data = request.json
        required_fields = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = [
            data['MDVP:Fo(Hz)'],
            data['MDVP:Fhi(Hz)'],
            data['MDVP:Flo(Hz)'],
            data['MDVP:Jitter(%)'],
            data['MDVP:Jitter(Abs)'],
            data['MDVP:RAP'],
            data['MDVP:PPQ'],
            data['Jitter:DDP'],
            data['MDVP:Shimmer'],
            data['MDVP:Shimmer(dB)'],
            data['Shimmer:APQ3'],
            data['Shimmer:APQ5'],
            data['MDVP:APQ'],
            data['Shimmer:DDA'],
            data['NHR'],
            data['HNR'],
            data['RPDE'],
            data['DFA'],
            data['spread1'],
            data['spread2'],
            data['D2'],
            data['PPE']
        ]

        prediction = parkinson_model.predict([input_data])
        result = "Parkinson's Disease" if prediction[0] == 1 else "No Parkinson's Disease"
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/lung_cancer', methods=['POST'])
def predict_lung_cancer():
    try:
        data = request.json
        required_fields = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONICDISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH', 'SWALLOWINGDIFFICULTY', 'CHESTPAIN']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = [
            data['GENDER'],
            data['AGE'],
            data['SMOKING'],
            data['YELLOW_FINGERS'],
            data['ANXIETY'],
            data['PEER_PRESSURE'],
            data['CHRONIC DISEASE'],
            data['FATIGUE'],
            data['ALLERGY'],
            data['WHEEZING'],
            data['ALCOHOL CONSUMING'],
            data['COUGHING'],
            data['SHORTNESS OF BREATH'],
            data['SWALLOWING DIFFICULTY'],
            data['CHEST PAIN']
        ]

        prediction = lung_cancer_model.predict([input_data])
        result = "Lung Cancer" if prediction[0] == 1 else "No Lung Cancer"
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/breast_cancer', methods=['POST'])
def predict_breast_cancer():
    try:
        data = request.json
        required_fields = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = [
            data['radius_mean'],
            data['texture_mean'],
            data['perimeter_mean'],
            data['area_mean'],
            data['smoothness_mean'],
            data['compactness_mean'],
            data['concavity_mean'],
            data['concave_points_mean'],
            data['symmetry_mean'],
            data['fractal_dimension_mean'],
            data['radius_se'],
            data['texture_se'],
            data['perimeter_se'],
            data['area_se'],
            data['smoothness_se'],
            data['compactness_se'],
            data['concavity_se'],
            data['concave_points_se'],
            data['symmetry_se'],
            data['fractal_dimension_se'],
            data['radius_worst'],
            data['texture_worst'],
            data['perimeter_worst'],
            data['area_worst'],
            data['smoothness_worst'],
            data['compactness_worst'],
            data['concavity_worst'],
            data['concave_points_worst'],
            data['symmetry_worst'],
            data['fractal_dimension_worst']
        ]

        prediction = breast_cancer_model.predict([input_data])
        result = "Breast Cancer" if prediction[0] == 1 else "No Breast Cancer"
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chronic_kidney', methods=['POST'])
def predict_chronic_kidney():
    try:
        data = request.json
        required_fields = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = [
            data['age'],
            data['bp'],
            data['sg'],
            data['al'],
            data['su'],
            data['rbc'],
            data['pc'],
            data['pcc'],
            data['ba'],
            data['bgr'],
            data['bu'],
            data['sc'],
            data['sod'],
            data['pot'],
            data['hemo'],
            data['pcv'],
            data['wc'],
            data['rc'],
            data['htn'],
            data['dm'],
            data['cad'],
            data['appet'],
            data['pe'],
            data['ane']
        ]

        prediction = chronic_disease_model.predict([input_data])
        result = "Chronic Kidney Disease" if prediction[0] == 1 else "No Chronic Kidney Disease"
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hepatitis', methods=['POST'])
def predict_hepatitis():
    try:
        data = request.json
        required_fields = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = [
            data['Age'],
            data['Sex'],
            data['ALB'],
            data['ALP'],
            data['ALT'],
            data['AST'],
            data['BIL'],
            data['CHE'],
            data['CHOL'],
            data['CREA'],
            data['GGT'],
            data['PROT']
        ]

        prediction = hepatitis_model.predict([input_data])
        result = "Hepatitis" if prediction[0] == 1 else "No Hepatitis"
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/liver', methods=['POST'])
def predict_liver():
    try:
        data = request.json
        required_fields = ['Sex', 'age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = [
            data['Sex'],
            data['age'],
            data['Total_Bilirubin'],
            data['Direct_Bilirubin'],
            data['Alkaline_Phosphotase'],
            data['Alamine_Aminotransferase'],
            data['Aspartate_Aminotransferase'],
            data['Total_Protiens'],
            data['Albumin'],
            data['Albumin_and_Globulin_Ratio']
        ]

        prediction = liver_model.predict([input_data])
        result = "Liver Disease" if prediction[0] == 1 else "No Liver Disease"
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0',port=5000)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
