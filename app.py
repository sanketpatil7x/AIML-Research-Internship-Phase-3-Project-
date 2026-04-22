import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# Load model & preprocessor
model = keras.models.load_model("models/optimized_model.h5")
preprocessor = joblib.load("preprocessor.pkl")

st.title("Loan Default Prediction App")

st.write("Enter applicant details:")

# Example inputs (customize based on your dataset)
income = st.number_input("Income", value=150000)
credit = st.number_input("Credit Amount", value=500000)
annuity = st.number_input("Annuity", value=20000)
age = st.number_input("Age (years)", value=30)
employment = st.number_input("Days Employed", value=2000)

# Create input dataframe
input_data = pd.DataFrame([{
    "AMT_INCOME_TOTAL": income,
    "AMT_CREDIT": credit,
    "AMT_ANNUITY": annuity,
    "DAYS_BIRTH": -age * 365,
    "DAYS_EMPLOYED": -employment
}])

# Fill missing columns (important)
for col in preprocessor.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[preprocessor.feature_names_in_]

# Predict
if st.button("Predict"):
    processed = preprocessor.transform(input_data)
    prob = model.predict(processed)[0][0]
    
    threshold = 0.6
    prediction = "Default" if prob > threshold else "No Default"
    
    st.subheader(f"Prediction: {prediction}")
    st.write(f"Confidence Score: {prob:.2f}")