import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model + scaler + columns
model = pickle.load(open("model/churn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

st.title("📊 Customer Churn Prediction System")

st.sidebar.header("Enter Customer Details")

# Inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 0, 150, 70)
total = st.sidebar.slider("Total Charges", 0, 10000, 1000)

# Predict
if st.sidebar.button("Predict Churn"):

    # Create input dictionary
    input_dict = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "InternetService": internet,
        "Contract": contract,
        "PaymentMethod": payment,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # One-hot encode input
    input_encoded = pd.get_dummies(input_df)

    # Add missing columns
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠ Customer is likely to CHURN!")
    else:
        st.success("✅ Customer is likely to STAY!")
