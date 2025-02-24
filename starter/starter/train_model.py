import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Import the necessary functions from your modules
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from ml.slice_performace import compute_slice_metrics, output_slice_metrics

# Load the data. Adjust the path as necessary.
data = pd.read_csv("../data/census_clean.csv")

# Split the data into training and test sets.
train_df, test_df = train_test_split(data, test_size=0.20, random_state=42)

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

# Process the training data.
X_train, y_train, encoder, lb = process_data(
    train_df,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Manually set the encoder's feature names so that later transformations work correctly.
encoder.feature_names_in_ = train_df[cat_features].columns

# Process the test data.
X_test, y_test, _, _ = process_data(
    test_df,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train the model.
model = train_model(X_train, y_train)

# Evaluate overall model performance.
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Overall Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F-beta: {fbeta:.4f}")

# --- Slice Analysis ---
# Pass the categorical features list to ensure the slice is processed identically to training.
slice_results = compute_slice_metrics(
    test_df,
    "education",
    model,
    encoder,
    lb,
    label_column="salary",
    categorical_features=cat_features
)
output_slice_metrics(slice_results, output_file="../model/slice_output.txt")
print("Slice performance metrics written to slice_output.txt.")

# Save the trained model and preprocessing artifacts.
joblib.dump(model, "../model/model.pkl")
joblib.dump(encoder, "../model/encoder.pkl")
joblib.dump(lb, "../model/lb.pkl")
print("Model training complete and artifacts saved.")