import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -------------------------------
# App config
# -------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction")
st.write("Predict whether a loan application will be **Approved** or **Rejected** using a Random Forest model.")

# -------------------------------
# Load model & encoders
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("feature_encoders.pkl", "rb") as f:
        feature_encoders = pickle.load(f)

    with open("target_encoder.pkl", "rb") as f:
        target_encoder = pickle.load(f)

    return model, feature_encoders, target_encoder


model, feature_encoders, target_encoder = load_artifacts()

# -------------------------------
# User input form
# -------------------------------
st.subheader("📋 Enter Applicant Details")

input_data = {}

with st.form("loan_form"):
    for feature in model.feature_names_in_:
        if feature in feature_encoders:
            # Categorical feature
            encoder = feature_encoders[feature]
            options = list(encoder.classes_)
            value = st.selectbox(feature.replace("_", " ").title(), options)
            input_data[feature] = value
        else:
            # Numerical feature
            value = st.number_input(
                feature.replace("_", " ").title(),
                min_value=0.0,
                step=1.0
            )
            input_data[feature] = value

    submitted = st.form_submit_button("🔮 Predict Loan Status")

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for col, encoder in feature_encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    # Predict
    prediction_encoded = model.predict(input_df)[0]
    prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]

    probabilities = model.predict_proba(input_df)[0]
    confidence = np.max(probabilities)

    # Display result
    st.subheader("📊 Prediction Result")

    if prediction_label.lower() == "approved":
        st.success(f"✅ Loan Approved")
    else:
        st.error(f"❌ Loan Rejected")

    st.write(f"**Confidence:** {confidence:.2%}")
