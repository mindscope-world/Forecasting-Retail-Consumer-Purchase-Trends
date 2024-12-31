import streamlit as st
import shap
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


# Define page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["XGBoost Multiclassification Model", "Feature Importance", "SHAP Visualization"])

# Model Page
if page == "XGBoost Multiclassification Model":
    st.title("XGBoost Multiclassification Model")
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


# Feature Importance Page
elif page == "Feature Importance":
    st.title("Feature Importance")
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

# SHAP Page
elif page == "SHAP Visualization":
    # Title for the Streamlit app
    st.title("SHAP Visualization for Model Interpretability")

    # Load the model and data
    @st.cache_resource
    def load_model_and_data():
        # Load the trained XGBoost model
        model = joblib.load("models/multi_classifier_xgb_reg.pkl")  # Replace with the actual path
        # Load the test dataset
        data = pd.read_csv("data/X_test.csv")  # Replace with the actual path
        return model, data

    # Initialize the model and data
    multi_classifier_xgb, X_test = load_model_and_data()

    # Ensure the test data matches the training data
    try:
        expected_features = multi_classifier_xgb.feature_names_in_  # Use `feature_names_in_` for sklearn-like models
        if expected_features is not None:
            X_test = X_test[expected_features]
        else:
            raise ValueError("Model does not have expected feature names. Check your model's training pipeline.")

        X_test = X_test[expected_features]
    except AttributeError:
        st.error("Model does not contain feature names. Ensure feature alignment manually.")

    # Preprocess data
    X_test = X_test.fillna(0).apply(pd.to_numeric, errors='coerce')

    # Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(multi_classifier_xgb.estimators_[0])  # Use one estimator for simplicity
    shap_values = explainer.shap_values(X_test)

    # Sidebar for user inputs
    st.sidebar.header("User Input")
    visualization_type = st.sidebar.selectbox(
        "Select Visualization Type",
        ["Summary Plot", "Force Plot"]
    )
    sample_idx = st.sidebar.number_input(
        "Select Sample Index for Force Plot",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1
    )

    # Main visualization area
    if visualization_type == "Summary Plot":
        st.subheader("Summary Plot: Global Interpretability")
        st.write("This plot shows the global impact of features on model predictions.")
        # Generate the SHAP summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
        st.pyplot(fig)

    elif visualization_type == "Force Plot":
        st.subheader("Force Plot: Local Interpretability")
        st.write(f"This plot explains the prediction for sample index {sample_idx}.")
        shap.initjs()
        # Generate the SHAP force plot

        try:
            force_plot_html = shap.force_plot(
                explainer.expected_value,
                shap_values[sample_idx],
                X_test.iloc[sample_idx],
                matplotlib=False,
                show=False
            )
            # Display the SHAP force plot in Streamlit
            st.components.v1.html(force_plot_html.html(), height=300)
        except Exception as e:
            st.error(f"Error generating force plot: {e}")

    # Debugging Information
    st.sidebar.header("Debugging Info")
    st.sidebar.write(f"Test data shape: {X_test.shape}")
    st.sidebar.write(f"SHAP values shape: {len(shap_values)}")