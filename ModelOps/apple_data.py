from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_apple_sales_data_with_promo_adjustment(
    base_demand: int = 1000,
    n_rows: int = 5000,
    feature_drift_factors: dict = None,
    concept_drift_factors: dict = None,
):
    """
    Generates a synthetic dataset for predicting apple sales demand with seasonality and inflation.

    This function creates a pandas DataFrame with features relevant to apple sales.
    The features include date, average_temperature, rainfall, weekend flag, holiday flag,
    promotional flag, price_per_kg, and the previous day's demand. The target variable,
    'demand', is generated based on a combination of these features with some added noise.

    Args:
        base_demand (int, optional): Base demand for apples. Defaults to 1000.
        n_rows (int, optional): Number of rows (days) of data to generate. Defaults to 5000.
        feature_drift_factors (dict, optional): Dictionary specifying drift for each feature.
            Example:
            {
                "average_temperature": 1.2,   # 20% increase
                "rainfall": 0.9,              # 10% decrease
                "price_per_kg": 1.09,         # 9% increase
                "promo": 1.5                  # 50% more frequent promotions
            }
        concept_drift_factors (dict, optional): Changes in feature-target relationships over time.
            {
                "price_sensitivity": 1.05,  # 5% increase in price sensitivity effect
                "promo_effect": 1.0,        # no change
                "weekend_effect": 1.15,     # 15% increase in weekend effect
                "feature_importance": True  # gradually change feature importance over time
            }
    Returns:
        pd.DataFrame: DataFrame with features and target variable for apple sales prediction.

    Example:
        >>> df = generate_apple_sales_data_with_seasonality(base_demand=1200, n_rows=6000)
        >>> df.head()
    """

    # Set seed for reproducibility
    np.random.seed(9999)

    # Set default drift factors if none are provided
    if feature_drift_factors is None:
        feature_drift_factors = {
            "average_temperature": 1.0,
            "rainfall": 1.0,
            "price_per_kg": 1.0,
            "promo": 1.0,
        }

    # Set default concept drift factors if none are provided
    if concept_drift_factors is None:
        concept_drift_factors = {
            "price_sensitivity": 1.0,  # No change initially
            "promo_effect": 1.0,
            "weekend_effect": 1.0,
            "feature_importance": False,
        }

    # Create date range
    dates = [datetime.now() - timedelta(days=i) for i in range(n_rows)]
    dates.reverse()

    # Generate features
    df = pd.DataFrame(
        {
            "date": dates,
            "average_temperature": np.random.uniform(10, 35, n_rows)
            * feature_drift_factors.get("average_temperature", 1.0),
            "rainfall": np.random.exponential(5, n_rows)
            * feature_drift_factors.get("rainfall", 1.0),
            "weekend": [(date.weekday() >= 5) * 1 for date in dates],
            "holiday": np.random.choice([0, 1], n_rows, p=[0.97, 0.03]),
            "price_per_kg": np.random.uniform(0.5, 3, n_rows)
            * feature_drift_factors.get("price_per_kg", 1.0),
            "month": [date.month for date in dates],
        }
    )

    # Introduce inflation over time (years)
    df["inflation_multiplier"] = (
        1 + (df["date"].dt.year - df["date"].dt.year.min()) * 0.03
    )

    # Incorporate seasonality due to apple harvests
    df["harvest_effect"] = np.sin(2 * np.pi * (df["month"] - 3) / 12) + np.sin(
        2 * np.pi * (df["month"] - 9) / 12
    )

    # Modify the price_per_kg based on harvest effect
    df["price_per_kg"] = df["price_per_kg"] - df["harvest_effect"] * 0.5

    # Adjust promo periods to coincide with periods lagging peak harvest by 1 month
    peak_months = [4, 10]  # months following the peak availability
    df["promo"] = np.where(
        df["month"].isin(peak_months),
        1,
        np.random.choice(
            [0, 1],
            n_rows,
            p=[
                1 - 0.15 * feature_drift_factors.get("promo", 1.0),
                0.15 * feature_drift_factors.get("promo", 1.0),
            ],
        ),
    )

    # Introduce **concept drift** by gradually changing feature importance over time
    if concept_drift_factors.get("feature_importance"):
        concept_shift = np.linspace(1, 1.0 - 0.4, n_rows)
    else:
        concept_shift = 1

    # Generate target variable based on features
    base_price_effect = (
        -df["price_per_kg"]
        * 50
        * concept_drift_factors.get("price_sensitivity", 1.0)
        * concept_shift
    )
    seasonality_effect = df["harvest_effect"] * 50
    promo_effect = (
        df["promo"]
        * 200
        * concept_drift_factors.get("promo_effect", 1.0)
        * concept_shift
    )
    weekend_effect = (
        df["weekend"]
        * 300
        * concept_drift_factors.get("weekend_effect", 1.0)
        * concept_shift
    )

    df["demand"] = (
        base_demand
        + base_price_effect
        + seasonality_effect
        + promo_effect
        + weekend_effect
        + np.random.normal(0, 50, n_rows)  # adding random noise
    ) * df["inflation_multiplier"]

    # convert to integer
    df["demand"] = df["demand"].round().astype(int)

    # Add previous day's demand
    df["previous_days_demand"] = df["demand"].shift(1)
    df["previous_days_demand"] = df[
        "previous_days_demand"
    ].bfill()  # fill the first row

    # Drop temporary columns
    df.drop(columns=["inflation_multiplier", "harvest_effect", "month"], inplace=True)

    return df
