import streamlit as st
import pandas as pd
from joblib import load

# Load the trained model
model = load('best_rf_model.joblib')

# Streamlit Form for user inputs
st.title("Heart Disease Prediction")

# Collect user inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ("Male", "Female"))
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=120)
exercise_habits = st.selectbox("Exercise Habits", ("Regular", "Occasional", "None"))
smoking = st.selectbox("Smoking", ("Yes", "No"))
family_heart_disease = st.selectbox("Family Heart Disease", ("Yes", "No"))
diabetes = st.selectbox("Diabetes", ("Yes", "No"))
bmi = st.number_input("BMI", min_value=10, max_value=50, value=22.0)
high_blood_pressure = st.selectbox("High Blood Pressure", ("Yes", "No"))
low_hdl_cholesterol = st.number_input("Low HDL Cholesterol (mg/dL)", min_value=0, max_value=100, value=40)
high_ldl_cholesterol = st.number_input("High LDL Cholesterol (mg/dL)", min_value=0, max_value=200, value=100)
alcohol_consumption = st.selectbox("Alcohol Consumption", ("Yes", "No"))
stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
sleep_hours = st.number_input("Sleep Hours per Night", min_value=0, max_value=24, value=7)
sugar_consumption = st.selectbox("Sugar Consumption", ("High", "Moderate", "Low"))
triglyceride_level = st.number_input("Triglyceride Level (mg/dL)", min_value=0, max_value=500, value=150)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar", ("Normal", "High"))
crp_level = st.number_input("CRP Level (mg/L)", min_value=0, max_value=10, value=1.0)
homocysteine_level = st.number_input("Homocysteine Level (µmol/L)", min_value=0, max_value=50, value=10.0)

# Convert gender, smoking, etc., to numerical values for model prediction
gender = 1 if gender == "Male" else 0
exercise_habits = 1 if exercise_habits == "Regular" else 0
smoking = 1 if smoking == "Yes" else 0
family_heart_disease = 1 if family_heart_disease == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
high_blood_pressure = 1 if high_blood_pressure == "Yes" else 0
alcohol_consumption = 1 if alcohol_consumption == "Yes" else 0
sugar_consumption = {"High": 2, "Moderate": 1, "Low": 0}[sugar_consumption]
fasting_blood_sugar = 1 if fasting_blood_sugar == "High" else 0

# Prepare the data in the same format as your model expects
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Blood Pressure': [blood_pressure],
    'Exercise Habits': [exercise_habits],
    'Smoking': [smoking],
    'Family Heart Disease': [family_heart_disease],
    'Diabetes': [diabetes],
    'BMI': [bmi],
    'High Blood Pressure': [high_blood_pressure],
    'Low HDL Cholesterol': [low_hdl_cholesterol],
    'High LDL Cholesterol': [high_ldl_cholesterol],
    'Alcohol Consumption': [alcohol_consumption],
    'Stress Level': [stress_level],
    'Sleep Hours': [sleep_hours],
    'Sugar Consumption': [sugar_consumption],
    'Triglyceride Level': [triglyceride_level],
    'Fasting Blood Sugar': [fasting_blood_sugar],
    'CRP Level': [crp_level],
    'Homocysteine Level': [homocysteine_level]
})

# Make predictions using the loaded model
if st.button('Predict Heart Disease'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]
    
    # Display the result
    if prediction == 1:
        st.write("Prediction: **High Risk** of Heart Disease")
    else:
        st.write("Prediction: **Low Risk** of Heart Disease")
    
    st.write("Prediction Probability:", prediction_proba[0])
