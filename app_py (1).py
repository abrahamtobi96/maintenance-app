-*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
import joblib
import streamlit as st
import pandas as pd
import numpy as np
# 
# # ---- Set page configuration (must be first Streamlit command) ----
# st.set_page_config(page_title='Maintenance Cost Prediction', page_icon=':bar_chart:', layout='wide')
# 
# # ---- Cache the model loading to avoid reloading every run ----
@st.cache_resource
def load_models():
mc_model_data = joblib.load('/content/mc_model.pkl')
dtf_model_data = joblib.load('/content/dtf_model.pkl')
# 
mc_model = mc_model_data['mc_model']           # Extract from dict
dtf_model = dtf_model_data['dtf_model_dt']     # Extract from dict
# 
return mc_model, dtf_model
# 
# # ---- Load Models ----
mc_model, dtf_model = load_models()
# 
# # ---- Page Title ----
st.markdown("<h1 style='text-align: center; color: red;'>Maintenance Cost Prediction</h1>", unsafe_allow_html=True)
st.subheader('Maintenance Cost Prediction')
# 
# # ---- Sidebar Input ----
st.sidebar.header('Equipment Parameters')
input_data = {
    'Temperature': st.sidebar.number_input("Temperature (°C)", min_value=0.00, max_value=100.00, value=50.00, help="Enter the equipment's operating temperature"),
    'Pressure': st.sidebar.number_input('Pressure (%)', min_value=0.00, max_value=100.00, value=50.00, help="Enter the equipment's operating pressure"),
    'Vibration': st.sidebar.number_input('Vibration', min_value=0.00, max_value=5.00, value=2.00, help="Enter the equipment's vibration"),
    'Humidity': st.sidebar.number_input('Humidity', min_value=0.00, max_value=2.00, value=1.00, help="Enter the equipment's humidity"),
    'Flow_Rate': st.sidebar.number_input('Flow Rate', min_value=0.00, max_value=15.00, value=10.00, help="Enter the equipment's flow rate"),
    'Power_Consumption': st.sidebar.number_input('Power Consumption', min_value=0.00, max_value=500.00, value=200.00, help="Enter the equipment's power consumption"),
    'Oil_Level': st.sidebar.number_input('Oil Level', min_value=0.00, max_value=1.00, value=0.50, help="Enter the equipment's oil level"),
    'Voltage': st.sidebar.number_input('Voltage', min_value=100.00, max_value=300.00, value=200.00, help="Enter the equipment's voltage"),
    'Production_Volume': st.sidebar.number_input('Production Volume', min_value=0, max_value=500, value=200, help="Enter the equipment's production volume"),
    'Planned_Downtime_Hours': st.sidebar.number_input('Planned Downtime Hours', min_value=0, max_value=24, value=8, help="Enter the equipment's planned downtime hours"),
    'Shifts_Per_Day': st.sidebar.number_input('Shifts Per Day', min_value=1, max_value=3, value=2, help="Enter the number of shifts per day"),
    'Production_Days_Per_Week': st.sidebar.number_input('Production Days Per Week', min_value=1, max_value=7, value=3, help="Enter the number of production days per week"),
    'Maintenance_Type_Corrective': st.sidebar.checkbox('Maintenance Type Corrective', help="Select if the maintenance type is corrective"),
    'Maintenance_Type_Preventive': st.sidebar.checkbox('Maintenance Type Preventive', help="Select if the maintenance type is preventive"),
    'Failure_Cause_Electrical_Failure': st.sidebar.checkbox('Failure Cause Electrical Failure', help="Select if the failure cause is electrical failure"),
    'Failure_Cause_Mechanical_Failure': st.sidebar.checkbox('Failure Cause Mechanical Failure', help="Select if the failure cause is mechanical failure"),
    'Failure_Cause_Sensor_Malfunction': st.sidebar.checkbox('Failure Cause Sensor Malfunction', help="Select if the failure cause is sensor malfunction"),

}
# 
# 
# ---- Preprocess Input ----
input_df = pd.DataFrame(input_data, index=[0])

# ---- Make Predictions ----
if st.button('Predict'):
    # Code to execute when the button is clicked
    # This is where you will call your prediction functions
    try:
        maintenance_cost = mc_model.predict(input_df)[0]
        days_till_failure = dtf_model.predict(input_df)[0]

        # Display Results
        st.success(f"Predicted Maintenance Cost: ${maintenance_cost:.2f}")
        st.warning(f"Predicted Days Till Failure: {days_till_failure:.0f} days")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.warning("Please check your input values and try again.")

# 
# ---- Custom Footer ----
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        padding: 10px;
    }
    .footer a {
        color: white;
        text-decoration: none;
        margin: 0 10px;
    }
    </style>
    <div class="footer">
        Developed by Tobi  - ©2025
        <br>
        <a href="https://www.linkedin.com/in/Tobi_Oluwasola/" target="_blank">LinkedIn</a>
        <a href="https://github.com/abrahamtobi96" target="_blank">GitHub</a>
    </div>
""", unsafe_allow_html=True)

