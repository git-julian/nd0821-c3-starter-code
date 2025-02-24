import requests

def main():
    # Replace this URL with the URL of your deployed API
    url = "https://nd0821-c3-starter-code-i4k3.onrender.com/inference"
    
    # Define the JSON payload that matches the InferenceInput schema
    payload = {
        "features": [2.0, 1.5, 1.0, 0.5, 1.0]
    }
    
    try:
        response = requests.post(url, json=payload)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()