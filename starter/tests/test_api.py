# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "message" in json_data
    assert json_data["message"] == "Welcome to the Census Income Prediction API!"

def test_post_predict_under_50k():
    sample = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert prediction in ["<=50K", ">50K"]

def test_post_predict_over_50k():
    sample = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 123456,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15000,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert prediction in ["<=50K", ">50K"]