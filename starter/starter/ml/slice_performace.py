import numpy as np
import pandas as pd
from ml.model import inference, compute_model_metrics

def compute_slice_metrics(df, feature, model, encoder, lb, label_column="salary", categorical_features=None):
    """
    Computes performance metrics (precision, recall, F-beta) for slices of the data
    based on a fixed value of a given categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
        The full DataFrame containing both features and label.
    feature : str
        The name of the categorical feature on which to slice.
    model : trained model
        The trained ML model used for inference.
    encoder : sklearn.preprocessing.TransformerMixin
        Encoder used to transform categorical features.
    lb : sklearn.preprocessing.LabelBinarizer or similar
        Binarizer used to transform the label.
    label_column : str, optional
        The name of the label column. Default is "salary".
    categorical_features : list, optional
        The list of categorical features used during training. If not provided,
        the encoder's stored feature names will be used.

    Returns
    -------
    slice_results : list of str
        A list of strings, each describing the performance metrics for one slice.
    """
    slice_results = []
    unique_values = df[feature].unique()
    
    # Use provided categorical_features or default to encoder's feature names
    if categorical_features is None:
        categorical_features = list(encoder.feature_names_in_)
    
    for value in unique_values:
        # Slice the data where the feature equals the current value
        df_slice = df[df[feature] == value]

        # Separate features and label
        y_slice = df_slice[label_column]
        df_features = df_slice.drop(columns=[label_column])
        
        # Separate categorical and continuous features
        X_cat = df_features[categorical_features]
        X_cont = df_features.drop(columns=categorical_features)
        
        # Transform categorical features using the fitted encoder
        X_cat_encoded = encoder.transform(X_cat)
        
        # Concatenate continuous features (if any) with the encoded categorical features.
        # This must match the processing done in process_data.
        if X_cont.shape[1] > 0:
            X_final = np.concatenate([X_cont.to_numpy(), X_cat_encoded], axis=1)
        else:
            X_final = X_cat_encoded

        # Transform the label
        y_slice_encoded = lb.transform(y_slice.values).ravel()

        # Run inference and compute metrics
        preds = inference(model, X_final)
        precision, recall, fbeta = compute_model_metrics(y_slice_encoded, preds)
        result_str = (f"Feature: {feature}, Value: {value} | "
                      f"Precision: {precision:.4f} | "
                      f"Recall: {recall:.4f} | "
                      f"F-beta: {fbeta:.4f}")
        slice_results.append(result_str)
    return slice_results

def output_slice_metrics(results, output_file="slice_output.txt"):
    """
    Writes the slice performance metrics to a text file.

    Parameters
    ----------
    results : list of str
        The list of strings describing slice metrics.
    output_file : str, optional
        The file name to which the metrics will be written. Default is "slice_output.txt".
    """
    with open(output_file, "w") as f:
        for line in results:
            f.write(line + "\n")