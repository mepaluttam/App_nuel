import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the trained model and features
model = joblib.load('delivery_time_prediction_model.pkl')

# Set page config
st.set_page_config(
    page_title="Delivery Time Prediction",
    page_icon="🚚",
    layout="wide"
)

# Add title and description
st.title("🚚 Delivery Time Prediction System")
st.markdown("""
This application predicts delivery times based on various input parameters.
Please fill in the details below to get a prediction.
""")

# Load the dataset for dropdown options
df = pd.read_csv("after_data_engineering.xls")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Route Information")

    # Numeric inputs
    distance_km = st.number_input("Distance (km)",
                                  min_value=0.0,
                                  max_value=10000.0,
                                  value=50.0)

    weight_kg = st.number_input("Weight (kg)",
                                min_value=0.0,
                                max_value=1000.0,
                                value=100.0)

    volume_m3 = st.number_input("Volume (m³)",
                                min_value=0.0,
                                max_value=100.0,
                                value=10.0)

    # City selection
    pickup_city = st.selectbox("Pickup City",
                               df["pickup_city"].unique())

    delivery_city = st.selectbox("Delivery City",
                                 df["delivery_city"].unique())

with col2:
    st.subheader("Conditions")

    # Weather and traffic inputs
    temperature = st.slider("Temperature (°C)",
                            min_value=-20,
                            max_value=50,
                            value=20)

    weather_main = st.selectbox("Weather Condition",
                                df["weather_main"].unique())

    traffic_multiplier = st.slider("Traffic Multiplier",
                                   min_value=1.0,
                                   max_value=3.0,
                                   value=1.0,
                                   step=0.1)

    traffic_level_category = st.selectbox("Traffic Level",
                                          ['Low', 'Medium', 'High'])

    # Delivery details
    priority = st.selectbox("Delivery Priority",
                            df["priority"].unique())

    vehicle_type = st.selectbox("Vehicle Type",
                                df["vehicle_type"].unique())

# Create a button to make prediction
if st.button("Predict Delivery Time"):
    try:
        # Create input dataframe with user inputs
        input_data = pd.DataFrame({
            'distance_km': [distance_km],
            'weight_kg': [weight_kg],
            'volume_m3': [volume_m3],
            'traffic_multiplier': [traffic_multiplier],
            'temperature': [temperature],
            'pickup_city': [pickup_city],
            'delivery_city': [delivery_city],
            'priority': [priority],
            'vehicle_type': [vehicle_type],
            'weather_main': [weather_main],
            'traffic_level_category': [traffic_level_category]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display results in a nice format
        st.success("### Prediction Results")

        # Create three columns for displaying results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Estimated Delivery Time",
                value=f"{prediction:.2f} hours"
            )

        with col2:
            st.metric(
                label="Estimated Delivery in Days",
                value=f"{prediction / 24:.1f} days"
            )

        with col3:
            # Calculate estimated delivery date/time
            current_time = datetime.now()
            delivery_time = current_time + pd.Timedelta(hours=prediction)
            st.metric(
                label="Estimated Delivery Date",
                value=delivery_time.strftime("%Y-%m-%d %H:%M")
            )

        # Add a map showing the delivery route (placeholder)
        st.subheader("Delivery Route")
        st.map()  # You can add actual coordinates for the route if available

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add footer with additional information
st.markdown("""
---
### About this Predictor
This model uses a Random Forest Regressor trained on historical delivery data. 
The prediction takes into account multiple factors including:
- Distance and route information
- Package characteristics
- Weather conditions
- Traffic patterns
- Vehicle type and priority level...
""")