from fastapi import FastAPI, HTTPException, UploadFile, File
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel
from typing import List
import io
import datetime
import json
import os

# Initialize FastAPI app
app = FastAPI()

# Set experiment name
mlflow.set_experiment("Forecasting Apple Demand")

# Load the trained model from MLflow
MODEL_URI = "models:/apple_demand@champion"  # Replace with your model name and alias
model = mlflow.pyfunc.load_model(MODEL_URI)

# Define the expected input schema for a single prediction
class InputData(BaseModel):
    average_temperature: float
    rainfall: float
    weekend: int
    holiday: int
    price_per_kg: float
    promo: int
    previous_days_demand: float

# Define log file path
LOG_FILE_PATH = "prediction_logs.csv"

# Ensure log file exists with headers
if not os.path.exists(LOG_FILE_PATH):
    pd.DataFrame(columns=[
        "timestamp", "request_type", "input_data", "predictions", "status"
    ]).to_csv(LOG_FILE_PATH, index=False)

def log_request(request_type, input_data, predictions, status):
    """Logs the request details to a CSV file."""
    log_entry = pd.DataFrame([{
        "timestamp": datetime.datetime.now().isoformat(),
        "request_type": request_type,
        "input_data": json.dumps(input_data),
        "predictions": json.dumps(predictions),
        "status": status
    }])
    log_entry.to_csv(LOG_FILE_PATH, mode='a', header=False, index=False)

@app.post("/predict")
def predict_single(input_data: List[InputData]):
    """Endpoint for real-time predictions with a single input."""
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict() for data in input_data])

        # Make predictions
        predictions = model.predict(df)

        # Log the request
        log_request("single", df.to_dict(orient="records"), predictions.tolist(), "success")

        return {"predictions": predictions.tolist()}
    except Exception as e:
        log_request("single", df.to_dict(orient="records"), None, f"error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Endpoint for batch predictions using a CSV file."""
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Validate required columns
        required_features = ["average_temperature", "rainfall", "weekend", "holiday", "price_per_kg", "promo", "previous_days_demand"]
        if not all(feature in df.columns for feature in required_features):
            missing_cols = set(required_features) - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

        # Make batch predictions
        predictions = model.predict(df)

        # Log the request
        log_request("batch", df.to_dict(orient="records"), predictions.tolist(), "success")

        return {"predictions": predictions.tolist()}
    except Exception as e:
        log_request("batch", df.to_dict(orient="records"), None, f"error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
