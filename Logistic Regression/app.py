import streamlit as st
import pandas as pd
import pickle
import os

# Page config
st.set_page_config(page_title="Diabetes Prediction App")

# Load model safely
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "diabetes_model.pkl")

    with open(model_path, "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("Diabetes Risk Prediction")
st.write("Enter patient clinical data to predict diabetes risk.")

# Inputs
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 3)
    glucose = st.number_input("Glucose", 0, 200, 117)
    blood_pressure = st.number_input("Blood Pressure", 0, 130, 72)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 23)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 30)
    bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.37)
    age = st.number_input("Age", 21, 100, 29)

# Prediction
if st.button("Predict Result"):
    input_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness,
          insulin, bmi, dpf, age]],
        columns=[
            "Pregnancies", "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"
        ]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"High Risk of Diabetes (Probability: {probability:.2%})")
    else:
        st.success(f"Low Risk of Diabetes (Probability: {probability:.2%})")
