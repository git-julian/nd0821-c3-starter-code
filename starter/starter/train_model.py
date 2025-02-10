# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Import from the starter code (adjust paths to your project structure).
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Load the data 
data = pd.read_csv("../data/census_clean.csv")  

# Optional enhancement: use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

# List of categorical features.
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

# Process the training data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train a model.
model = train_model(X_train, y_train)

# Evaluate the model on the test set.
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F-beta: {fbeta:.4f}")

# Save the trained model and the encoder objects.
joblib.dump(model, "../model/model.pkl")
joblib.dump(encoder, "../model/encoder.pkl")
joblib.dump(lb, "../model/lb.pkl")

print("Model training complete and artifacts saved.")