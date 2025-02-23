# test_api.py
from fastapi.testclient import TestClient
from starter.main import app  

client = TestClient(app)

def test_get_endpoint():
    """
    Test the GET endpoint:
    - It should return a 200 status code.
    - It should return the expected greeting message.
    """
    response = client.get("/")
    assert response.status_code == 200, "GET request did not return status code 200."
    json_data = response.json()
    # Check that the response contains the expected message.
    assert "message" in json_data, "Response JSON does not contain 'message' key."
    assert json_data["message"] == "Welcome to the Census Income Inference API!", "Greeting message is incorrect."

def test_inference_output_zero():
    """
    Test the POST /inference endpoint for a case where the inference should return 0.
    The dummy inference returns 0 if the sum of features is not greater than 5.0.
    """
    # This payload sums to 0.9 which is less than the threshold 5.0.
    payload = {"features": [0.1, 0.2, 0.3, 0.1, 0.2]}
    response = client.post("/inference", json=payload)
    assert response.status_code == 200, "POST request did not return status code 200."
    json_data = response.json()
    assert "prediction" in json_data, "Response JSON does not contain 'prediction' key."
    assert json_data["prediction"] == 0, "Expected prediction of 0 for sum below threshold."

def test_inference_output_one():
    """
    Test the POST /inference endpoint for a case where the inference should return 1.
    The dummy inference returns 1 if the sum of features is greater than 5.0.
    """
    # This payload sums to 6.0 which is greater than the threshold 5.0.
    payload = {"features": [2.0, 1.5, 1.0, 0.5, 1.0]}
    response = client.post("/inference", json=payload)
    assert response.status_code == 200, "POST request did not return status code 200."
    json_data = response.json()
    assert "prediction" in json_data, "Response JSON does not contain 'prediction' key."
    assert json_data["prediction"] == 1, "Expected prediction of 1 for sum above threshold."