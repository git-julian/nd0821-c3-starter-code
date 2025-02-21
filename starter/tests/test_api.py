from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_root():
    """
    Test the GET endpoint to verify status code and the returned greeting.
    """
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "message" in json_data
    assert json_data["message"] == "Welcome to the Census Income Inference API!"

def test_inference_output_zero():
    """
    Test the POST endpoint when the input features produce a sum below threshold,
    expecting a prediction of 0.
    """
    # Input that sums to less than 5.0: e.g., [0.1, 0.2, 0.3, 0.1, 0.2] (total 0.9)
    input_data = {"features": [0.1, 0.2, 0.3, 0.1, 0.2]}
    response = client.post("/inference", json=input_data)
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert json_data["prediction"] == 0

def test_inference_output_one():
    """
    Test the POST endpoint when the input features produce a sum above threshold,
    expecting a prediction of 1.
    """
    # Input that sums to more than 5.0: e.g., [2.0, 1.5, 1.0, 0.5, 1.0] (total 6.0)
    input_data = {"features": [2.0, 1.5, 1.0, 0.5, 1.0]}
    response = client.post("/inference", json=input_data)
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert json_data["prediction"] == 1