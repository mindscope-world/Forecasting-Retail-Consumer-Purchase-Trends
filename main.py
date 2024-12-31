import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier

# Load Data and Preprocessing
@st.cache_data
def load_data():
    # Replace with actual data loading
    data = pd.read_csv("data/data.csv")  # Example file name
    encoded_data = pd.read_csv("data/encoded.csv")
    return data, encoded_data

data, encoded_data = load_data()

# Define mappings for columns
reverse_column_mappings = {
    # Add mappings for each column (example below)
    'Gender': {0: 'Male', 1: 'Female'},
    'Marital-status': {0: 'Married', 1: 'Single', 2: 'Divorced', 3: 'Cohabiting'},
    # ... Add all column mappings here
}

column_mappings = {col: {v: k for k, v in reverse_column_mappings[col].items()} for col in reverse_column_mappings}

# Splitting Data
feature_columns = ["Gender", "Age-group", "Income-Level", "Occupation"]  # Add all your feature columns here
target_columns = ["Factor 1", "Factor 2", "Factor 3", "Factor 4"]  # Replace with actual targets

# 1. Preprocess the data
# Define the target variables (multi-label targets)
y = encoded_data[['Gender', 'Age-group', 'Occupation', "Income-Level"]]

# Define feature variables (exclude target columns and irrelevant ones)
X = data.drop(columns=['Gender', 'Age-group', 'Occupation', 'Income-Level', 'Year', 'Month'])

# Encode categorical variables in features
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train MultiOutput XGBoost Model
@st.cache_data
def train_model():
    base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    multi_classifier_xgb = MultiOutputClassifier(base_model)
    multi_classifier_xgb.fit(X_train, y_train)
    return multi_classifier_xgb

multi_classifier_xgb = train_model()

# Sidebar for Inputs
st.sidebar.header("Input Parameters")
user_input = {}
for col in feature_columns:
    unique_values = data[col].unique()
    user_input[col] = st.sidebar.selectbox(f"Select {col}", unique_values)

# Button for Prediction
if st.sidebar.button("Submit"):
    # Convert user input to numerical
    input_data = pd.DataFrame([user_input])
    for col, mapping in column_mappings.items():
        if col in input_data.columns:
            input_data[col] = input_data[col].map(mapping)

    # Make Predictions
    predictions = multi_classifier_xgb.predict(input_data)

    # Decode Predictions
    decoded_predictions = {}
    for col, val in zip(target_columns, predictions[0]):
        decoded_predictions[col] = reverse_column_mappings[col].get(val, f"Unknown value ({val})")

    # Display Predictions
    st.subheader("Predictions")
    st.table(pd.DataFrame([decoded_predictions]))

# Feature Importance Visualization
st.sidebar.header("Feature Importance Visualization")
feature_importance_data = {}
for i, estimator in enumerate(multi_classifier_xgb.estimators_):
    feature_importances = estimator.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]
    feature_importance_data[f"Output {i + 1}"] = {
        "features": [X_train.columns[idx] for idx in sorted_idx],
        "importances": feature_importances[sorted_idx]
    }

selected_output = st.sidebar.selectbox("Select Output for Feature Importance", options=list(feature_importance_data.keys()))

if selected_output:
    st.subheader(f"Feature Importance for {selected_output}")
    data = feature_importance_data[selected_output]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(data["features"], data["importances"])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance for {selected_output}")
    ax.invert_yaxis()
    st.pyplot(fig)
