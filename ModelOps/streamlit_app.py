import streamlit as st
import pandas as pd
import requests

# FastAPI backend URL. Used to send requests from Streamlit
# to FastAPI for model predictions.
FASTAPI_URL = "http://127.0.0.1:5000"

st.title("Apple Demand Forecasting - ML Model UI")

#################################
# === Single Input Prediction ===
#################################
# Users can manually enter feature values for each feature
st.subheader("Single Input Prediction")
demand_date = st.date_input("Date Prediction is For")
average_temperature = st.number_input("Average Temperature (°C)", value=28.5)
rainfall = st.number_input("Rainfall (mm)", value=1.4)
weekend = st.selectbox("Is it a weekend?", [0, 1])
holiday = st.selectbox("Is it a holiday?", [0, 1])
price_per_kg = st.number_input("Price per Kg ($)", value=1.54)
promo = st.selectbox("Promotion Available?", [0, 1])
previous_days_demand = st.number_input("Previous Days Demand", value=1313)

# This button triggers a request to FastAPI's /predict endpoint.
if st.button("Predict Demand"):
    input_data = [{
        "demand_date": demand_date,
        "average_temperature": average_temperature,
        "rainfall": rainfall,
        "weekend": weekend,
        "holiday": holiday,
        "price_per_kg": price_per_kg,
        "promo": promo,
        "previous_days_demand": previous_days_demand
    }]

    response = requests.post(f"{FASTAPI_URL}/predict", json=input_data)

    # The response is displayed on the Streamlit UI.
    if response.status_code == 200:
        prediction = response.json()["predictions"][0]
        st.success(f"Predicted Demand: {prediction:.2f}")
    else:
        st.error("Error fetching prediction. Check FastAPI logs.")

###########################################
# === Batch Prediction with File Upload ===
###########################################
# Users can upload a CSV file containing multiple rows of input data.
st.subheader("Batch Prediction via CSV Upload")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())

    # The file is sent to the FastAPI /predict_batch endpoint.
    if st.button("Get Batch Predictions"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{FASTAPI_URL}/predict_batch", files=files)

        #Predictions are added to the dataset and displayed on the UI.
        if response.status_code == 200:
            predictions = response.json()["predictions"]
            df["Predicted Demand"] = predictions
            st.subheader("Predictions:")
            st.write(df)
        else:
            st.error("Error processing batch prediction.")
