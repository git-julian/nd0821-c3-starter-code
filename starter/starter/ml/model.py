from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    # Example: Using Random Forest as the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
        Precision of the model.
    recall : float
        Recall of the model.
    fbeta : float
        F-beta score of the model.
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ 
    Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


# Example usage (replace X_train, y_train, X_test, y_test with your actual data):
if __name__ == "__main__":
    # Example data (you should replace this with actual training/testing data)
    X_train = np.random.rand(100, 5)  # 100 samples, 5 features
    y_train = np.random.randint(0, 2, 100)  # Binary classification labels

    # Train the model
    model = train_model(X_train, y_train)

    # Example test data
    X_test = np.random.rand(20, 5)  # 20 samples, 5 features
    y_test = np.random.randint(0, 2, 20)

    # Run inference
    predictions = inference(model, X_test)

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)

    print(f"Precision: {precision}, Recall: {recall}, F-beta: {fbeta}")