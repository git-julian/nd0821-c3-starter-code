# test_model.py
import os
import pytest
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

# Import from your own modules:
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

# ----- FIXTURES -----

@pytest.fixture(scope="session")
def data():
    """
    Fixture to load the Census dataset.
    """
    csv_path = os.path.join("..","data", "census_clean.csv")  
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV at {csv_path}")
    df = pd.read_csv(csv_path)
    return df


@pytest.fixture(scope="session")
def processed_data(data):
    """
    Fixture to process the dataset, splitting into X, y with the provided cat features.
    Returns X, y, encoder, lb.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(
        data, 
        categorical_features=cat_features, 
        label="salary", 
        training=True
    )
    return X, y, encoder, lb

# ----- TESTS -----

def test_data_shape(data):
    """
    Ensure the imported DataFrame is not empty.
    """
    assert data.shape[0] > 0, "Loaded data has 0 rows."
    assert data.shape[1] > 0, "Loaded data has 0 columns."

def test_process_data_return_types(processed_data):
    """
    Test that process_data returns the expected types and shapes.
    """
    X, y, encoder, lb = processed_data
    assert isinstance(X, np.ndarray), "X should be a NumPy array."
    assert isinstance(y, np.ndarray), "y should be a NumPy array."
    assert isinstance(encoder, OneHotEncoder), "encoder should be OneHotEncoder."
    assert isinstance(lb, LabelBinarizer), "lb should be a LabelBinarizer."
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples."

def test_train_model(processed_data):
    """
    Train a model on the processed data and verify the model type and a sample inference.
    """
    X, y, encoder, lb = processed_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier."

    # Test that we can run inference on a small subset
    sample_X = X[:5]
    preds = inference(model, sample_X)
    assert len(preds) == 5, "Number of predictions should match the number of samples."

def test_model_metrics(processed_data):
    """
    Check that compute_model_metrics gives valid floats for precision, recall, fbeta.
    """
    X, y, _, _ = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)

    # Check type
    for metric in [precision, recall, fbeta]:
        assert isinstance(metric, float), "Metric should be a float."
    # Check range
    assert 0.0 <= precision <= 1.0, "Precision out of range [0, 1]."
    assert 0.0 <= recall <= 1.0,    "Recall out of range [0, 1]."
    assert 0.0 <= fbeta <= 1.0,    "F-beta out of range [0, 1]."

def test_save_and_load_model(processed_data):
    """
    Optionally, test saving/loading the model with joblib, if you want
    to confirm pipeline artifact handling.
    """
    X, y, _, _ = processed_data
    model = train_model(X, y)

    # Save model
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "test_model.pkl")
    joblib.dump(model, model_path)

    # Load model
    loaded_model = joblib.load(model_path)
    # Check inference with loaded model
    preds = inference(loaded_model, X[:5])
    assert len(preds) == 5, "Loaded model inference failed."

    # Clean up (if desired)
    if os.path.exists(model_path):
        os.remove(model_path)
    # Optionally remove the directory if empty
    try:
        os.rmdir("model")
    except OSError:
        pass