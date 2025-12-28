import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

import warnings
warnings.filterwarnings("ignore")

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Andrew2505/Tourism-package-project", filename="best_tourism_package_model.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourist package Prediction
st.title("Tourism Package Prediction App")
st.write("This app predicts whether customers are likely to take a tourism package.")
st.write("Kindly enter the customer details below:")

# Collect user input
Age = st.number_input("Age", min_value=18, max_value=61, value=36)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=2)
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=5, value=2)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=4)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips", min_value=1, max_value=22, value=3)
Passport = st.number_input("Passport", min_value=0, max_value=1, value=1)
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
Designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional", "Other"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=5000)

# Convert Yes/No inputs to numeric
def yes_no_to_int(value):
    return 1 if value == "Yes" else 0

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Age": int(Age),
    "TypeofContact": TypeofContact,
    "CityTier": int(CityTier),
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": int(NumberOfPersonVisiting),
    "PreferredPropertyStar": int(PreferredPropertyStar),
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": int(NumberOfTrips),
    "Passport": yes_no_to_int(Passport),
    "OwnCar": yes_no_to_int(OwnCar),
    "Designation": Designation,
    "MonthlyIncome": float(MonthlyIncome)
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Taken Package" if prediction == 1 else "not taken package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
