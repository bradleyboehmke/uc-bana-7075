import io
import os
from typing import List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Retrieve MLflow tracking URI from environment variable
mlflow_tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI", "file:///app/mlflow_registry/mlruns"
)
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Set MLflow experiment name
mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "Forecasting Apple Demand")
mlflow.set_experiment(mlflow_experiment)

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


@app.post("/predict")
def predict_single(input_data: List[InputData]):
    """Endpoint for real-time predictions with a single input."""

    # Convert input to DataFrame
    df = pd.DataFrame([data.dict() for data in input_data])

    try:
        # Make predictions
        predictions = model.predict(df)

        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Endpoint for batch predictions using a CSV file."""
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Validate required columns
        required_features = [
            "average_temperature",
            "rainfall",
            "weekend",
            "holiday",
            "price_per_kg",
            "promo",
            "previous_days_demand",
        ]
        if not all(feature in df.columns for feature in required_features):
            missing_cols = set(required_features) - set(df.columns)
            raise HTTPException(
                status_code=400, detail=f"Missing columns: {missing_cols}"
            )

        # Make batch predictions
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
