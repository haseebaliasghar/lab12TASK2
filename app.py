import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn  # IMPORTANT: ensures pickle can find scikit-learn classes

# ================== APP TITLE ==================
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")
st.title("💰 Loan Approval Prediction App")
st.write("Enter applicant details to predict loan approval status using a trained Random Forest model.")

# ================== LOAD MODEL ==================
@st.cache_data
def load_model():
    # Explicitly import sklearn here to avoid pickle errors
    import sklearn
    with open("random_forest_loan_approval_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("loan_label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

rf_model, label_encoders = load_model()

# ================== USER INPUT ==================
st.sidebar.header("Applicant Details")

def user_input():
    input_data = {}
    for feature in rf_model.feature_names_in_:
        if feature in label_encoders:
            # Categorical input
            options = list(label_encoders[feature].classes_)
            input_data[feature] = st.sidebar.selectbox(f"{feature}:", options)
        else:
            # Numerical input
            input_data[feature] = st.sidebar.number_input(f"{feature}:", value=0)
    return pd.DataFrame([input_data])

input_df = user_input()

# ================== ENCODE CATEGORICALS ==================
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

# ================== PREDICTION ==================
prediction = rf_model.predict(input_df)[0]
prediction_prob = rf_model.predict_proba(input_df)[0]

# ================== DISPLAY RESULTS ==================
st.subheader("Prediction")
if prediction == 1:
    st.success("✅ Loan Approved")
else:
    st.error("❌ Loan Not Approved")

st.subheader("Prediction Probabilities")
prob_df = pd.DataFrame({
    "Status": label_encoders['loan_status'].classes_,
    "Probability": prediction_prob
})
st.dataframe(prob_df)

# ================== FEATURE IMPORTANCE ==================
st.subheader("Feature Importance")
import matplotlib.pyplot as plt
import seaborn as sns

importances = rf_model.feature_importances_
features = rf_model.feature_names_in_

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x=importances, y=features, ax=ax, palette="Purples")
ax.set_title("Random Forest Feature Importance")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
st.pyplot(fig)
