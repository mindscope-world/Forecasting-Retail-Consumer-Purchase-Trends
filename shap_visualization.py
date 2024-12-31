import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt
import joblib

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
