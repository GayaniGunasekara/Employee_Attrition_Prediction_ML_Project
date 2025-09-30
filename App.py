import streamlit as st
import pandas as pd
import joblib

# --- Load trained pipeline ---
pipeline = joblib.load("attrition_pipeline.pkl")

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
    "Predict whether an employee is likely to **leave the company** based on HR features."
)

# --- Sidebar Information ---
st.sidebar.header("About App")
st.sidebar.info(
    """
    Welcome to the Employee Attrition Predictor!!!  

    This app uses a smart Machine Learning model to analyze  
    employee details and estimate the chance of leaving the company.  

     Simply fill in the form with employee information  
    and click "Predict" to see the result instantly..
    """
)

# --- Input Widgets ---
st.header("📋 Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=60, step=1, value=30)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Education = st.selectbox(
        "Education Level", [1, 2, 3, 4, 5],
        help="1: Below College, 2: College, 3: Bachelor, 4: Master, 5: Doctor"
    )

with col2:
    BusinessTravel = st.selectbox(
        "Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )
    Department = st.selectbox(
        "Department", ["Sales", "Research & Development", "Human Resources"]
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
            "Technician"
        ],
    )

MonthlyIncome = st.number_input(
    "Monthly Income", min_value=1000, max_value=50000, step=500, value=5000
)
DistanceFromHome = st.slider("Distance From Home (km)", 1, 50, 5)

# --- Convert inputs into DataFrame (raw, no manual encoding) ---
user_input = pd.DataFrame([{
    "Age": Age,
    "BusinessTravel": BusinessTravel,
    "Department": Department,
    "JobRole": JobRole,
    "Gender": Gender,
    "MonthlyIncome": MonthlyIncome,
    "DistanceFromHome": DistanceFromHome,
    "Education": Education
}])

# --- Predict ---
if st.button("🔍 Predict Attrition"):
    # probability of Attrition = 1
    prob = pipeline.predict_proba(user_input)[0][1]
    prediction = "Likely to Leave 😟" if prob > 0.5 else "Likely to Stay 🙂"

    st.subheader("🔎 Prediction Result")
    st.metric(label="Attrition Probability", value=f"{prob*100:.2f}%")

    if prob > 0.5:
        st.error(prediction)
    else:
        st.success(prediction)
