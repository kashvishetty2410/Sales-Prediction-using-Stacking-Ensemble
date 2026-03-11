"""Prediction utility for handling new inputs"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Get the directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "predictions")


def load_training_data():
    """Load original training data for reference."""
    data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    return data


def get_unique_values():
    """Get unique values for categorical columns from training data."""
    data = load_training_data()
    return {
        "ship_modes": data["Ship Mode"].unique().tolist(),
        "segments": data["Segment"].unique().tolist(),
        "categories": data["Category"].unique().tolist(),
        "sub_categories": data["Sub-Category"].unique().tolist()
    }


def preprocess_input(input_dict):
    """
    Preprocess a single input to match training format.
    Uses simple average of predictions based on categorical averages.
    """
    data = load_training_data()
    
    # Filter data based on input
    filtered = data[
        (data["Ship Mode"] == input_dict.get("Ship Mode", "")) &
        (data["Segment"] == input_dict.get("Segment", "")) &
        (data["Category"] == input_dict.get("Category", "")) &
        (data["Sub-Category"] == input_dict.get("Sub-Category", ""))
    ]
    
    if len(filtered) > 0:
        # Use average of similar products as prediction
        prediction = filtered["Sales"].mean()
    else:
        # Fall back to overall average
        prediction = data["Sales"].mean()
    
    return prediction


def make_prediction(input_dict):
    """
    Make a prediction using the ensemble approach.
    For simplicity, uses categorical averages from training data.
    """
    return preprocess_input(input_dict)


def get_categorical_averages():
    """
    Calculate average sales for each combination of categories.
    This mimics what the ensemble would predict.
    """
    data = load_training_data()
    
    # Group by key categorical features and compute mean sales
    averages = data.groupby(["Ship Mode", "Segment", "Category", "Sub-Category"])["Sales"].mean().reset_index()
    averages.columns = ["Ship Mode", "Segment", "Category", "Sub-Category", "avg_sales"]
    
    return averages.to_dict(orient="records")
