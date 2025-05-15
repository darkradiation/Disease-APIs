from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Constants for General Disease Prediction Model ---
# These lists should ideally be loaded from files generated during the model training phase.
# ALL_SYMPTOMS_LIST: Alphabetically sorted list of all 133 unique symptoms used as features.
ALL_SYMPTOMS_LIST = [
  "abdominal_pain",
  "abnormal_menstruation",
  "acidity",
  "acute_liver_failure",
  "altered_sensorium",
  "anxiety",
  "back_pain",
  "belly_pain",
  "blackheads",
  "bladder_discomfort",
  "blister",
  "blood_in_sputum",
  "bloody_stool",
  "blurred_and_distorted_vision",
  "breathlessness",
  "brittle_nails",
  "bruising",
  "burning_micturition",
  "chest_pain",
  "chills",
  "cold_hands_and_feets",
  "coma",
  "congestion",
  "constipation",
  "continuous_feel_of_urine",
  "continuous_sneezing",
  "cough",
  "cramps",
  "dark_urine",
  "dehydration",
  "depression",
  "diarrhoea",
  "dischromic _patches",
  "distention_of_abdomen",
  "dizziness",
  "drying_and_tingling_lips",
  "enlarged_thyroid",
  "excessive_hunger",
  "extra_marital_contacts",
  "family_history",
  "fast_heart_rate",
  "fatigue",
  "fluid_overload",
  "foul_smell_of urine",
  "headache",
  "high_fever",
  "hip_joint_pain",
  "history_of_alcohol_consumption",
  "increased_appetite",
  "indigestion",
  "inflammatory_nails",
  "internal_itching",
  "irregular_sugar_level",
  "irritability",
  "irritation_in_anus",
  "itching",
  "joint_pain",
  "knee_pain",
  "lack_of_concentration",
  "lethargy",
  "loss_of_appetite",
  "loss_of_balance",
  "loss_of_smell",
  "loss_of_taste",
  "malaise",
  "mild_fever",
  "mood_swings",
  "movement_stiffness",
  "mucoid_sputum",
  "muscle_pain",
  "muscle_wasting",
  "muscle_weakness",
  "nausea",
  "neck_pain",
  "nodal_skin_eruptions",
  "obesity",
  "pain_behind_the_eyes",
  "pain_during_bowel_movements",
  "pain_in_anal_region",
  "painful_walking",
  "palpitations",
  "passage_of_gases",
  "patches_in_throat",
  "phlegm",
  "polyuria",
  "prominent_veins_on_calf",
  "puffy_face_and_eyes",
  "pus_filled_pimples",
  "receiving_blood_transfusion",
  "receiving_unsterile_injections",
  "red_sore_around_nose",
  "red_spots_over_body",
  "redness_of_eyes",
  "restlessness",
  "runny_nose",
  "rusty_sputum",
  "scurring",
  "shivering",
  "silver_like_dusting",
  "sinus_pressure",
  "skin_peeling",
  "skin_rash",
  "slurred_speech",
  "small_dents_in_nails",
  "spinning_movements",
  "spotting_ urination",
  "stiff_neck",
  "stomach_bleeding",
  "stomach_pain",
  "sunken_eyes",
  "sweating",
  "swelled_lymph_nodes",
  "swelling_joints",
  "swelling_of_stomach",
  "swollen_blood_vessels",
  "swollen_extremeties",
  "swollen_legs",
  "throat_irritation",
  "tiredness",
  "toxic_look_(typhos)",
  "ulcers_on_tongue",
  "unsteadiness",
  "visual_disturbances",
  "vomiting",
  "watering_from_eyes",
  "weakness_in_limbs",
  "weakness_of_one_body_side",
  "weight_gain",
  "weight_loss",
  "yellow_crust_ooze",
  "yellow_urine",
  "yellowing_of_eyes",
  "yellowish_skin",
];

# DISEASE_CLASSES: Alphabetically sorted list of 42 unique disease names (target classes).
DISEASE_CLASSES = [
    '(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne',
    'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma',
    'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
    'Common Cold', 'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)',
    'Drug Reaction', 'Fungal infection', 'GERD', 'Gastroenteritis',
    'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
    'Hepatitis E', 'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia',
    'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine',
    'Osteoarthristis', 'Paralysis (brain hemorrhage)',
    'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis',
    'Typhoid', 'Urinary tract infection', 'Varicose veins', 'hepatitis A'
]

# Create a mapping from symptom name to index for faster lookups
SYMPTOM_TO_INDEX = {symptom: i for i, symptom in enumerate(ALL_SYMPTOMS_LIST)}

# Load all the pre-trained models
diabetes_model = joblib.load("models/diabetes_prediction_decision_tree.sav")  # recall --> 0.709
heart_model = joblib.load("models/heart_disease_prediction_svm.sav")  # recall --> 0.875
breast_cancer_model = joblib.load("models/breast_cancer_prediction_adb.sav") # recall --> 0.953
hepatitis_model = joblib.load("models/hepatitis_c_prediction_xgb.sav") # recall --> 0.958
parkinson_model = joblib.load("models/parkinsons_disease_prediction_rf.sav")  # recall--> 1.00
liver_model = joblib.load("models/liver_disease_prediction_svm.sav")  # recall--> 1.00
chronic_kidney_disease_model = joblib.load("models/chronic_kidney_disease_prediction_rf.sav") # recall--> 1.00
lung_cancer_model = joblib.load("models/lung_cancer_prediction_xgb.sav") # recall--> 1.00
general_disease_model = joblib.load("models/general_disease_prediction_xgb.sav") # recall--> 1.0000


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
        required_fields = ['Gender', 'Age', 'Smoking', 'Yellow_Fingers', 'Anxiety', 'Peer_Pressure', 'Chronic_Disease', 'Fatigue', 'Allergy', 'Wheezing', 'Alcohol_Consuming', 'Coughing', 'Shortness_Of_Breath', 'Swallowing_Difficulty', 'Chest_Pain']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = [
            data['Gender'],
            data['Age'],
            data['Smoking'],
            data['Yellow_Fingers'],
            data['Anxiety'],
            data['Peer_Pressure'],
            data['Chronic_Disease'],
            data['Fatigue'],
            data['Allergy'],
            data['Wheezing'],
            data['Alcohol_Consuming'],
            data['Coughing'],
            data['Shortness_Of_Breath'],
            data['Swallowing_Difficulty'],
            data['Chest_Pain']
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
        required_fields = ['Bp', 'Sg', 'Al', 'Su', 'Rbc', 'Bu', 'Sc', 'Sod', 'Pot', 'Hemo', 'Wbcc', 'Rbcc', 'Htn']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        input_data = [
            data['Bp'],
            data['Sg'],
            data['Al'],
            data['Su'],
            data['Rbc'],
            data['Bu'],
            data['Sc'],
            data['Sod'],
            data['Pot'],
            data['Hemo'],
            data['Wbcc'],
            data['Rbcc'],
            data['Htn']
        ]

        prediction = chronic_kidney_disease_model.predict([input_data])
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
        result = "Hepatitis" if prediction[0] == 1 else "No Hepatitis" # Assuming 1 is Hepatitis, 0 is No Hepatitis
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

@app.route('/api/general_disease', methods=['POST'])
def predict_general_disease():
    try:
        data = request.json
        if 'symptoms' not in data or not isinstance(data['symptoms'], list):
            return jsonify({'error': 'Missing or invalid field: symptoms (must be a list of strings)'}), 400

        user_symptoms = data['symptoms']

        # Create the 133-feature input vector (binary encoded)
        input_feature_vector = [0] * len(ALL_SYMPTOMS_LIST)

        for symptom in user_symptoms:
            if not isinstance(symptom, str):
                return jsonify({'error': 'Invalid symptom format: all symptoms must be strings'}), 400
            
            symptom_clean = symptom.strip() # Clean symptom as done in training
            if symptom_clean in SYMPTOM_TO_INDEX:
                input_feature_vector[SYMPTOM_TO_INDEX[symptom_clean]] = 1
            # Optionally, log or handle unknown symptoms if needed
            # else:
            #     print(f"Warning: Unknown symptom '{symptom_clean}' provided by user.")

        prediction_encoded = general_disease_model.predict([input_feature_vector])
        
        # Map encoded prediction back to disease name
        # Ensure prediction_encoded[0] is a valid index for DISEASE_CLASSES
        if 0 <= prediction_encoded[0] < len(DISEASE_CLASSES):
            predicted_disease_name = DISEASE_CLASSES[prediction_encoded[0]]
        else:
            return jsonify({'error': 'Model prediction out of bounds for known diseases'}), 500
            
        return jsonify({'result': predicted_disease_name})

    except Exception as e:
        # Log the full exception for debugging on the server
        app.logger.error(f"Error in /api/general_disease: {str(e)}", exc_info=True)
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0',port=5000)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))