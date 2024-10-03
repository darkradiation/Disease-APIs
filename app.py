from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained diabetes model
diabetes_model = joblib.load("models/diabetes_model.sav")

@app.route('/api/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Get the data from the POST request
        data = request.json

        # Ensure all required fields are present
        required_fields = ['pregnancies', 'glucose', 'bloodPressure', 'skinThickness', 'insulin', 'bmi', 'diabetesPedigreeFunction', 'age']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Prepare the data for prediction
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

        # Make the prediction
        prediction = diabetes_model.predict([input_data])

        # Return the result
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
