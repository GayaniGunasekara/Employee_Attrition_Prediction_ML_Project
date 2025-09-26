import streamlit as st
import pandas as pd
import pickle

# --- Load trained model and column names ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# --- Your Streamlit widgets and prediction code below ---
st.title("Employee Attrition Prediction")

# Example input
# --- User input widgets ---
Age = st.number_input("Age", min_value=18, max_value=60)
BusinessTravel = st.selectbox(
    "Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
Department = st.selectbox(
    "Department", ["Sales", "Research & Development", "HR"])

# --- Convert inputs into DataFrame and fix missing columns ---
input_dict = {
    "Age": Age,
    "BusinessTravel_Travel_Frequently": 1 if BusinessTravel == "Travel_Frequently" else 0,
    "BusinessTravel_Travel_Rarely": 1 if BusinessTravel == "Travel_Rarely" else 0,
    "BusinessTravel_Non-Travel": 1 if BusinessTravel == "Non-Travel" else 0,
    "Department_Research & Development": 1 if Department == "Research & Development" else 0,
    "Department_Sales": 1 if Department == "Sales" else 0,
    "Department_HR": 1 if Department == "HR" else 0,
}

input_df = pd.DataFrame([input_dict])

# Add missing columns with 0
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training
input_df = input_df[model_columns]


prob = model.predict_proba(input_df)[:, 1][0]
st.write(f"Probability of leaving: {prob*100:.2f}%")
