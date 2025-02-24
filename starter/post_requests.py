import requests

def main():
    # URL of your deployed API's inference endpoint
    url = "https://nd0821-c3-starter-code-i4k3.onrender.com/inference"
    
    # Provided detailed census data
    census_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    
    # Extract a list of features in the order expected by the model.
    # Adjust the order and which fields to include as needed.
    features = [
        census_data["age"],
        census_data["education_num"],
        census_data["fnlgt"],
        census_data["capital_gain"],
        census_data["hours_per_week"]
    ]
    
    # Create the payload with the expected key "features"
    payload = {"features": features}
    
    try:
        response = requests.post(url, json=payload)
        print("Status Code:", response.status_code)
        try:
            print("Response:", response.json())
        except ValueError:
            print("Response is not valid JSON. Raw response:")
            print(response.text)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()