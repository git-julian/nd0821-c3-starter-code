import requests

base_url = "https://<your-service-name>.onrender.com"  # update with your actual URL

# Test GET endpoint
get_response = requests.get(base_url)
print("GET Response:", get_response.status_code, get_response.json())

# Test POST endpoint
payload = {"features": [2.0, 1.5, 1.0, 0.5, 1.0]}
post_response = requests.post(f"{base_url}/inference", json=payload)
print("POST Response:", post_response.status_code, post_response.json())