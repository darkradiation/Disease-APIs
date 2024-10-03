import requests

def test_diabetes_api():
    url = "http://127.0.0.1:5000/api/diabetes"
    data = {
        "pregnancies": 6,
        "glucose": 148,
        "bloodPressure": 72,
        "skinThickness": 35,
        "insulin": 0,
        "bmi": 33.6,
        "diabetesPedigreeFunction": 0.627,
        "age": 50
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['result']}")
    else:
        print(f"Error: {response.status_code}, {response.json()['error']}")

if __name__ == "__main__":
    test_diabetes_api()