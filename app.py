import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, and columns

model = joblib.load("diabetes_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction System")
st.markdown("Predict diabetes risk using a trained Machine Learning model.")


# User Inputs

st.header("Patient Information")

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)

hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

smoking_history = st.selectbox(
    "Smoking History",
    ["never", "No Info", "former", "current", "ever", "not current"]
)

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=120)

# Encoding

gender_map = {"Female": 0, "Male": 1}
smoking_map = {
    "No Info": 0,
    "current": 1,
    "ever": 2,
    "former": 3,
    "never": 4,
    "not current": 5
}

input_dict = {
    "gender": gender_map[gender],
    "age": age,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "smoking_history": smoking_map[smoking_history],
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": glucose
}


# Create DataFrame with correct column order

input_df = pd.DataFrame([input_dict])

# Ensure column order matches training
input_df = input_df.reindex(columns=model_columns)


# Scale numerical features

num_features = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
input_df[num_features] = scaler.transform(input_df[num_features])


# Prediction

if st.button("Predict Diabetes"):
    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1]
    else:
        probability = None

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    if probability is not None:
        st.info(f"Prediction Probability: **{probability:.2%}**")




