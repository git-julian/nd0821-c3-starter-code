# Model Card for Census Income Classification

## Model Details
- **Model Name:** Census Income Classification Model
- **Version:** 1.0
- **Date:** 2025-02-21
- **Developed by:** Julian
- **Framework:** Scikit-learn (RandomForestClassifier)

## Intended Use
This model is designed to predict whether an individual's income exceeds $50K per year based on census data. It is intended for research and exploratory analysis purposes, and should not be used as the sole basis for financial or hiring decisions.

## Overview
The model is a binary classifier built using a Random Forest algorithm. It leverages features such as age, workclass, education, marital status, occupation, relationship, race, sex, and native-country to make predictions about an individual's income category (">50K" or "<=50K").

## Model Architecture
- **Algorithm:** Random Forest Classifier
- **Key Hyperparameters:**
  - `n_estimators`: 100
  - `random_state`: 42

## Training Data
- **Data Source:** Census Income Dataset (e.g., UCI Machine Learning Repository)
- **Number of Samples:** Approximately 32,000 training samples (80% split) and 8,000 testing samples (20% split)
- **Preprocessing:** Categorical features are one-hot encoded, and the target is binarized.

## Evaluation Data
- The model is evaluated on a 20% hold-out test set.
- Additional slice-based metrics lsare computed for subgroups defined by the `education` feature to assess model performance across different demographic segments.

## Metrics
- **Precision:** Measures the proportion of positive predictions that are correct.
- **Recall:** Measures the proportion of actual positives that are correctly identified.
- **F-beta (F1 Score):** The harmonic mean of precision and recall.
- **Slice Metrics:** Performance metrics (precision, recall, and F-beta) are also computed on slices of data based on the `education` feature. For example, for each unique education level, separate metrics are reported to help understand the model's behavior across subgroups.

### Example Performance Metrics
- **Overall Performance:**
  - Precision: 0.7391
  - Recall: 0.6384
  - F1 Score: 0.6851
- **Slice Performance:** See `slice_output.txt` for detailed metrics on each education level.

## Limitations
- The model's performance may vary across different demographic subgroups.
- There is potential for bias if the training data is not representative of the current population.
- The model may underperform on groups with limited training examples.

## Ethical Considerations
- The model is intended to assist in analysis and should not be used in isolation for decision-making.
- Users should consider potential biases in the data and model when interpreting predictions.
- Continuous monitoring and updating of the model are recommended to mitigate bias and drift.

## Caveats and Recommendations
- **Interpretability:** Random Forest models can be challenging to interpret; feature importance scores can help provide insights.
- **Updating:** Retraining the model with newer data is recommended to maintain accuracy.
- **Deployment:** The model should be used alongside domain expertise and human oversight.

## Contact Information
For more information or questions regarding this model, please contact [Your Contact Information].