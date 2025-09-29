import streamlit as st
import pandas as pd
import pickle

# --- Load trained model and columns ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# --- Page Config ---
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🧑‍💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1.5em;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #27ae60;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title("🧑‍💼 Employee Attrition Prediction App")
st.markdown(
    "Predict whether an employee is likely to **leave the company** based on HR features.")

# --- Sidebar Information ---
st.sidebar.header("About App")
st.sidebar.info(
    """
    📊 This app uses a Machine Learning model (Logistic Regression)  
    trained on HR Employee Attrition data.  
    Enter employee details below to get prediction.
    """
)

# --- Input Widgets ---
st.header("📋 Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=60, step=1)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Education = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                             help="1: Below College, 2: College, 3: Bachelor, 4: Master, 5: Doctor")

with col2:
    BusinessTravel = st.selectbox(
        "Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )
    Department = st.selectbox(
        "Department", ["Sales", "Research & Development", "HR"]
    )
    JobRole = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Human Resources",
            "Technical Architect"
        ],
    )

MonthlyIncome = st.number_input(
    "Monthly Income", min_value=1000, max_value=50000, step=500)
DistanceFromHome = st.slider("Distance From Home (km)", 1, 50, 5)

# --- Convert inputs into DataFrame ---
input_dict = {
    "Age": Age,
    "MonthlyIncome": MonthlyIncome,
    "DistanceFromHome": DistanceFromHome,
    "Education": Education,

    # One-hot encode BusinessTravel
    "BusinessTravel_Travel_Frequently": 1 if BusinessTravel == "Travel_Frequently" else 0,
    "BusinessTravel_Travel_Rarely": 1 if BusinessTravel == "Travel_Rarely" else 0,
    "BusinessTravel_Non-Travel": 1 if BusinessTravel == "Non-Travel" else 0,

    # One-hot encode Department
    "Department_Research & Development": 1 if Department == "Research & Development" else 0,
    "Department_Sales": 1 if Department == "Sales" else 0,
    "Department_HR": 1 if Department == "HR" else 0,

    # One-hot encode Gender
    "Gender_Male": 1 if Gender == "Male" else 0,
    "Gender_Female": 1 if Gender == "Female" else 0,
}

# One-hot encode JobRole
job_roles = [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Human Resources", "Technical Architect"
]
for role in job_roles:
    input_dict[f"JobRole_{role}"] = 1 if JobRole == role else 0

input_df = pd.DataFrame([input_dict])

# --- Handle missing columns ---
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder to match training
input_df = input_df[model_columns]

# --- Predict ---
if st.button("🔍 Predict Attrition"):
    prob = model.predict_proba(input_df)[:, 1][0]
    prediction = "Likely to Leave 😟" if prob > 0.5 else "Likely to Stay 🙂"

    st.subheader("🔎 Prediction Result")
    st.metric(label="Attrition Probability", value=f"{prob*100:.2f}%")
    st.success(prediction if prob < 0.5 else prediction)
