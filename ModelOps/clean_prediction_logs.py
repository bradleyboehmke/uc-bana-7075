import json

import pandas as pd


def clean_prediction_logs(prediction_logs):
    """Cleans and processes prediction logs by separating single and batch
    requests, expanding nested JSON structures, and combining the results
    into a single DataFrame.

    Args:
        prediction_logs (pd.DataFrame): DataFrame containing prediction logs
                                        with columns 'request_type', 'input_data',
                                        and 'predictions'.

    Returns:
        pd.DataFrame: Cleaned and processed DataFrame with expanded feature columns
                      and predictions.
    """
    # Create a dataframe for single requests & batch requests
    singles_df = prediction_logs[prediction_logs["request_type"] == "single"]
    batch_df = prediction_logs[prediction_logs["request_type"] == "batch"]

    # Create JSON structure for nested input_data & predictions columns
    batch_df["input_data"] = batch_df["input_data"].apply(json.loads)
    batch_df["predictions"] = batch_df["predictions"].apply(json.loads)

    # Expand input_data column and convert input_data dictionary into
    # separate feature columns
    batch_df_input = batch_df.explode("input_data")
    batch_features = pd.json_normalize(batch_df_input["input_data"])

    # Expand predictions column and join with feature data for a final
    # cleaned log of batch data
    batch_predictions = batch_df.explode("predictions", ignore_index=True)
    batch_logs = batch_predictions.join(batch_features).drop(columns="input_data")

    # Now rinse and repeate with the single, real-time requests
    singles_df["input_data"] = singles_df["input_data"].apply(json.loads)
    singles_df["predictions"] = singles_df["predictions"].apply(json.loads)
    singles_df = singles_df.explode("predictions")
    singles_input = singles_df.explode("input_data")
    singles_logs = singles_df.join(pd.json_normalize(singles_input["input_data"])).drop(
        columns="input_data"
    )

    combined_logs = pd.concat([singles_logs, batch_logs], ignore_index=True)
    combined_logs.rename(columns={"timestamp": "date"}, inplace=True)
    return combined_logs
