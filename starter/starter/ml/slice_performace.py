import pandas as pd
from ml.model import inference, compute_model_metrics

def compute_slice_metrics(df, feature, model, encoder, lb, label_column="salary"):
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

    Returns
    -------
    slice_results : list of str
        A list of strings, each describing the performance metrics for one slice.
    """
    slice_results = []
    unique_values = df[feature].unique()
    for value in unique_values:
        # Slice the data where the feature is equal to the current value
        df_slice = df[df[feature] == value]

        # Separate features and label
        X_slice = df_slice.drop(columns=[label_column])
        y_slice = df_slice[label_column]

        # Transform features and label (using the provided encoder and label binarizer)
        X_slice_encoded = encoder.transform(X_slice)
        y_slice_encoded = lb.transform(y_slice.values).ravel()

        # Run inference and compute metrics
        preds = inference(model, X_slice_encoded)
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