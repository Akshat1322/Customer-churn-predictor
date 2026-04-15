import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── LOAD PIPELINE ───────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    with open("models/churn_pipeline.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifacts = load_pipeline()
    pipe = artifacts["pipeline"]
    explainer = artifacts["explainer"]
except:
    st.error("Run train.py first to generate model")
    st.stop()

model = pipe.named_steps["model"]
preprocessor = pipe.named_steps["preprocessor"]

# ── HEADER ─────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction System")
st.caption("Pipeline + SHAP Explainability")
st.markdown("---")

# ── INPUT FORM ─────────────────────────────────────────────
st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 18, 120, 65)
    total_charges = st.slider("Total Charges", 0, 9000, int(tenure * monthly_charges))
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# ── PREDICTION ─────────────────────────────────────────────
if st.button("Predict Churn"):

    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_dict])

    # Prediction using pipeline
    prob = pipe.predict_proba(input_df)[0][1]

    # Threshold
    threshold = 0.3
    pred = int(prob > threshold)

    # Display result
    if prob > 0.7:
        st.error(f"🔴 High Risk: {prob:.2%}")
    elif prob > 0.4:
        st.warning(f"🟡 Medium Risk: {prob:.2%}")
    else:
        st.success(f"🟢 Low Risk: {prob:.2%}")

    # ── SHAP EXPLANATION ───────────────────────────────────
    st.subheader("Why this prediction?")

    X_transformed = preprocessor.transform(input_df)
    shap_values = explainer.shap_values(X_transformed)

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_transformed[0]
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()

# ── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.caption("Built by Akshat · End-to-End ML System")