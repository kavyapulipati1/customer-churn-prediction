import streamlit as st
import numpy as np
import pickle

# Load model + scaler + encoders
model = pickle.load(open("model/churn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
encoders = pickle.load(open("model/encoders.pkl", "rb"))

st.title("📊 Customer Churn Prediction System")

st.sidebar.header("Enter Customer Details")

# -----------------------------
# Input Fields (All Features)
# -----------------------------

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

MonthlyCharges = st.sidebar.slider("Monthly Charges", 0, 150, 70)
TotalCharges = st.sidebar.slider("Total Charges", 0, 10000, 1000)

# -----------------------------
# Encode Inputs Properly
# -----------------------------
def encode_input(col, value):
    return encoders[col].transform([value])[0]

if st.sidebar.button("Predict Churn"):

    input_dict = {
        "gender": encode_input("gender", gender),
        "Partner": encode_input("Partner", Partner),
        "Dependents": encode_input("Dependents", Dependents),
        "tenure": tenure,
        "PhoneService": encode_input("PhoneService", PhoneService),
        "InternetService": encode_input("InternetService", InternetService),
        "Contract": encode_input("Contract", Contract),
        "PaymentMethod": encode_input("PaymentMethod", PaymentMethod),
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }

    # ⚠ Fill missing columns with default 0
    full_features = np.zeros((1, 19))

    # Put known values into correct positions
    feature_list = list(encoders.keys()) + ["tenure", "MonthlyCharges", "TotalCharges"]

    i = 0
    for col in feature_list:
        if col in input_dict:
            full_features[0, i] = input_dict[col]
        i += 1

    # Scale
    scaled_input = scaler.transform(full_features)

    # Predict
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("⚠ Customer is likely to CHURN!")
    else:
        st.success("✅ Customer is likely to STAY!")
