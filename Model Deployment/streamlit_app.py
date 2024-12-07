#imports
import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import os

# Initialize model
model = None

# Load the trained model
filename = os.path.join(os.path.dirname(__file__), 'ridge_regression_model.sav')

# Check if the file exists
if os.path.exists(filename):
    st.write(f"Model file found at {filename}.")
    try:
        with open(filename, 'rb') as model_file:
            model = pickle.load(model_file)
            st.success("Model loaded successfully!")
    except Exception as e:
        model = None
        st.error(f"Error loading the model: {str(e)}")
else:
    st.error("Model file not found. Check the file path!")

# Define the function to make predictions
def predict_yield(input_data):
    if model is None:
        raise ValueError("Model not loaded properly.")
    input_data = np.array(input_data).reshape(1, -1)  # Reshape the input for prediction
    try:
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Streamlit app interface
st.title("Maize Yield Prediction in Kenya")
st.write("Enter the climate parameters to predict maize yield:")

# Create 2 columns for layout
col1, col2 = st.columns(2)

# Column 1 inputs with sliders
with col1:
    temperature_2m = st.slider("Temperature 2m (°C)", min_value=24.0, max_value=26.0, value=24.0)
    temperature_2m_max = st.slider("Max Temperature 2m (°C)", min_value=31.0, max_value=33.0, value=31.0)
    temperature_2m_min = st.slider("Min Temperature 2m (°C)", min_value=18.0, max_value=20.0, value=18.0)

# Column 2 inputs with sliders
with col2:
    total_precipitation_sum = st.slider("Total Precipitation (mm)", min_value=409.0, max_value=991.0, value=409.0)
    u_component_of_wind_10m = st.slider("U Component of Wind 10m (m/s)", min_value=-2.1, max_value=-1.3, value=-2.1)
    precipitation = st.slider("Precipitation (mm)", min_value=443.0, max_value=971.0, value=443.0)

# Button to make predictions
if st.button("Predict Maize Yield"):
    input_data = [
        temperature_2m, 
        temperature_2m_max, 
        temperature_2m_min, 
        total_precipitation_sum, 
        u_component_of_wind_10m, 
        precipitation
    ]

    # Log inputs for debugging (optional)
    st.write(f"Input data: {input_data}")
    
    # Call the predict function
    try:
        predicted_yield = predict_yield(input_data)
        # Determine the yield quality
        if predicted_yield < 1.2:
            yield_quality = "Bad"
        elif predicted_yield > 1.7:
            yield_quality = "Good"
        else:
            yield_quality = "Normal"

        # Display the predicted yield and quality
        st.success(f"Predicted Maize Yield: {predicted_yield:.2f} MT/HA")
        st.write(f"Yield: **{yield_quality}**")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
