import streamlit as st
import pickle
import numpy as np

# Load the trained Model and Scaler
model = pickle.load(open('cardio_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Cardiovascular Risk Predictor", layout="centered")

st.title("❤️ Cardio Health Risk Predictor")
st.write("Enter your health details below to predict the risk of cardiovascular disease:")

# UI Inputs in English
age = st.number_input("Age (in Years)", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ["Female", "Male"])
height = st.number_input("Height (in cm)", min_value=50, max_value=250, value=165)
weight = st.number_input("Weight (in kg)", min_value=10, max_value=300, value=70)
ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=50, max_value=250, value=120)
ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=30, max_value=150, value=80)

cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
gluc = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])

smoke = st.selectbox("Do you smoke?", ["No", "Yes"])
alco = st.selectbox("Do you consume alcohol?", ["No", "Yes"])
active = st.selectbox("Are you physically active?", ["Yes", "No"])

# Values Mapping
gender_val = 1 if gender == "Female" else 2
chol_val = 1 if cholesterol == "Normal" else (2 if cholesterol == "Above Normal" else 3)
gluc_val = 1 if gluc == "Normal" else (2 if gluc == "Above Normal" else 3)
smoke_val = 1 if smoke == "Yes" else 0
alco_val = 1 if alco == "Yes" else 0
active_val = 1 if active == "Yes" else 0

if st.button("Predict"):
    # Features List: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
    features = np.array([[age, gender_val, height, weight, ap_hi, ap_lo, chol_val, gluc_val, smoke_val, alco_val, active_val]])

    # Standardize inputs
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)

    st.write("---")
    if prediction[0] == 1:
        st.error("⚠️ **Warning:** There is a high risk of cardiovascular disease. Please consult a doctor.")
    else:
        st.success("✅ **Good News:** The risk of cardiovascular disease is very low! Keep maintaining a healthy lifestyle.")
