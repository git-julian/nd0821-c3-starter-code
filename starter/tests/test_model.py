# tests/test_model.py
import pandas as pd
import numpy as np
from starter.train_model import process_data

def test_process_data_output():
    # Create a dummy DataFrame.
    df = pd.DataFrame({
        "age": [25, 30],
        "workclass": ["Private", "Self-emp"],
        "fnlgt": [226802, 200000],
        "education": ["HS-grad", "Bachelors"],
        "education-num": [9, 13],
        "marital-status": ["Never-married", "Married"],
        "occupation": ["Adm-clerical", "Tech-support"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "capital-gain": [0, 500],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 45],
        "native-country": ["United-States", "United-States"],
        "income": ["<=50K", ">50K"]
    })
    categorical_features = ["workclass", "education", "marital-status", 
                            "occupation", "relationship", "race", "sex", "native-country"]
    X, y, encoder = process_data(df, categorical_features, label="income", training=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    # Check that the encoder is fitted.
    assert hasattr(encoder, "categories_")