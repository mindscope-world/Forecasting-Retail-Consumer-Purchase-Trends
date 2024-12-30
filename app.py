import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Save feature_columns and target_columns (add this block to your notebook/script for saving)
def save_columns(feature_columns, target_columns):
    joblib.dump(feature_columns, 'feature_columns.pkl')
    joblib.dump(target_columns, 'target_columns.pkl')

# Load the trained XGBoost model
# Assuming the model and column names are saved as files
multi_classifier_xgb = joblib.load('models/multi_classifier_xgb_reg.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
target_columns = joblib.load('models/target_columns.pkl')

# Column mappings
column_mappings = {
    'Period': {v: i for i, v in enumerate(['20/03/2024', '20/11/2023', '20/07/2023', '20/03/2023', '20/11/2022',
                                           '20/06/2022', '20/02/2022', '20/10/2021', '20/07/2021', '20/04/2021',
                                           '20/12/2020', '20/11/2020', '20/10/2020', '20/09/2020'])},
    'Location': {v: i for i, v in enumerate(['N36', 'N4', 'N12', 'N20', 'N28', 'N11', 'N25', 'N2', 'NK1', 'N14', 'N8', 'N17',
                                              'N16', 'N21', 'N27', 'N47', 'N41', 'N42', 'N5', 'N48', 'N9', 'N35', 'N50',
                                              'N18', 'N31', 'N13', 'N6', 'N22', 'N30', 'N10', 'N1', 'MS1', 'N43', 'N34',
                                              'N23', 'N37', 'D7', 'AC30', 'N40', 'N45', 'N46', 'D39', 'Z9', 'M1', 'N24',
                                              'N19', 'AC28', 'N15', 'L47', 'N29', 'T18', 'D3', 'N26', 'N7', 'N38', 'AB2',
                                              'L32', 'N3', 'N33', 'T11', 'D44', 'D16', 'D22', 'D21', 'D23', 'D43', 'D41',
                                              'D15', 'D40', 'T23', 'D42', 'AB7', 'AB5', 'N39'])},
    'Gender': {'Male': 0, 'Female': 1},
    'Marital-status': {'Married': 0, 'Cohabiting': 1, 'Dating': 2, 'Divorced/Separated/Widowed': 3, 'Single': 4,
                       'Married ': 0, 'Single ': 4, 'Dating ': 2},
    'Age-group': {'Millennials': 0, 'Gen X': 1, 'Gen Z': 2, 'Baby Boomers': 3},
    'Occupation': {'Business owner': 0, 'Salaried employee': 1, 'Self employed/Contractor': 2, 'Unemployed': 3,
                   'Student': 4, 'Stay-at-home mom or dad': 5, 'Salaried employee ': 1, 'Business owner ': 0,
                   'Student ': 4, 'Unemployed ': 3},
    'Income-Level': {'Middle Income': 0, 'Low Income': 1, 'High Income': 2},
    ' Personal-Hygiene (eg Soap, toothpaste)': {'More': 0, 'Same': 1, 'Less': 2, 'not_sure': 3},
    'Cleaning-products ': {'More': 0, 'Same': 1, 'Less': 2, 'not_sure': 3},
    'Fresh-groceries (Fruits, vegetables)': {'More': 0, 'Same': 1, 'Less': 2, 'not_sure': 3},
    'Medicines-Natural-remedies': {'Same': 0, 'Less': 1, 'not_sure': 2, 'More': 3},
    'Alcohol-beverages': {'More': 0, 'Same': 1, 'not_sure': 2, 'Less': 3},
    'Skin-care (eg. Body lotion)': {'More': 0, 'Same': 1, 'not_sure': 2, 'Less': 3},
    'Hair-care (eg. Shampoo)': {'More': 0, 'Same': 1, 'not_sure': 2, 'Less': 3},
    'Entertainment (eg. Restaurants, movies)': {'More': 0, 'Same': 1, 'Less': 2, 'not_sure': 3},
    'Beauty (eg Makeup, cosmetics, haircuts)': {'More': 0, 'Same': 1, 'Less': 2, 'not_sure': 3},
    'Clothing': {'More': 0, 'Same': 1, 'Less': 2, 'not_sure': 3},
    'Airtime-Data-bundles': {'Same': 0, 'More': 1, 'Less': 2, 'not_sure': 3}
}

# Reverse mappings for predictions
reverse_column_mappings = {col: {v: k for k, v in mapping.items()} for col, mapping in column_mappings.items()}

# Sidebar for user input
st.sidebar.header("Input Features")
user_input = {}
selected_values = {}
for feature in feature_columns:
    if feature in column_mappings:
        options = list(column_mappings[feature].keys())
        selected_value = st.sidebar.selectbox(f"{feature}", options)
        selected_values[feature] = selected_value
        user_input[feature] = column_mappings[feature][selected_value]
    else:
        user_input[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# Submit button
if st.sidebar.button("Submit"):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Perform prediction
    predictions = multi_classifier_xgb.predict(input_df)

    # Map predictions back to their string equivalents with error handling
    decoded_predictions = {}
    for col, val in zip(target_columns, predictions[0]):
        try:
            decoded_predictions[col] = reverse_column_mappings[col][val]
        except KeyError:
            decoded_predictions[col] = f"Unknown value ({val})"

    # Display selected independent values
    st.write("### Selected Independent Values")
    st.table(pd.DataFrame([selected_values]))

    # Display the predictions
    st.write("### Predictions")
    st.table(pd.DataFrame([decoded_predictions]))
