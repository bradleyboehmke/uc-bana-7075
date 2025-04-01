import json

import mlflow
import pandas as pd
from clean_prediction_logs import clean_prediction_logs
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.report import Report

###################################
# MLFlow Experiment Configuration #
###################################
# Set MLflow experiment
mlflow.set_experiment("Forecasting Apple Demand")

# Define model location
MODEL_URI = "models:/apple_demand@champion"

# Load the trained ML model
model = mlflow.pyfunc.load_model(MODEL_URI)

# Retrieve the run ID of the logged model
model_run_id = model.metadata.run_id

print("Model successfully loaded.")

################
# Prepare Data #
################
# Load data model was trained on
training_data_path = f"training_data/{model_run_id}-training_data.csv"
training_data = pd.read_csv(training_data_path)
training_data['predictions'] = model.predict(training_data)

print("Training data successfully loaded.")

# Load recent data with predictions and clean
log_file_path = "prediction_logs.csv"
df = pd.read_csv(log_file_path)
cleaned_logs = clean_prediction_logs(df)
# Retain columns in cleaned_logs that are features in training_data
cols = training_data.drop(columns="demand").columns.to_list()
cleaned_logs = cleaned_logs[cols]

print("Prediction logs successfully loaded and cleaned.")

##########################
# Generate Drift Reports #
##########################
# Define column mapping
column_mapping = ColumnMapping()
column_mapping.target = "demand"
column_mapping.prediction = "predictions"
column_mapping.datetime = "date"

# Generate Data Drift Report for feature columns only
feature_columns = training_data.drop(columns=["demand", "predictions"]).columns.to_list()
data_drift_report = Report(metrics=[DataDriftPreset(drift_share=0.3)])
data_drift_report.run(
    reference_data=training_data[feature_columns],
    current_data=cleaned_logs[feature_columns],
    column_mapping=ColumnMapping()
)
data_drift_report.save_html("data_drift_report.html")

print("Feature drift report generated successfully.")

# Generate Model Performance Report
performance_report = Report(metrics=[RegressionPreset()])
performance_report.run(
    reference_data=training_data,
    current_data=cleaned_logs,
    column_mapping=column_mapping,
)
performance_report.save_html("performance_report.html")

print("Model performance report generated successfully.")
