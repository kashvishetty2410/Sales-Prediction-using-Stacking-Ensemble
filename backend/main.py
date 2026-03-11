"""FastAPI backend for Sales Prediction"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.prediction import make_prediction, get_unique_values

app = FastAPI(title="Sales Prediction API", description="API for predicting product sales using stacking ensemble")


class PredictionInput(BaseModel):
    ship_mode: str
    segment: str
    category: str
    sub_category: str


class PredictionOutput(BaseModel):
    predicted_sales: float
    model: str = "Stacking Ensemble (Average)"


@app.get("/")
def root():
    return {"message": "Sales Prediction API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/options")
def get_prediction_options():
    """Return available options for each input field."""
    options = get_unique_values()
    return options


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Predict sales based on input features.
    """
    try:
        # Create input dictionary
        input_dict = {
            'Ship Mode': input_data.ship_mode,
            'Segment': input_data.segment,
            'Category': input_data.category,
            'Sub-Category': input_data.sub_category
        }
        
        # Make prediction
        prediction = make_prediction(input_dict)
        
        return PredictionOutput(predicted_sales=round(prediction, 2))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/performance")
def get_model_performance():
    """Return performance metrics for each base model."""
    return {
        "models": {
            "Linear Regression": {"r2": 0.09},
            "Random Forest": {"r2": -0.01},
            "SVR": {"r2": -0.03}
        },
        "stacking": {"r2": 0.092}
    }
