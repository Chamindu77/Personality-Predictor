"""
Endpoints:
  GET  /         → Welcome message
  GET  /health   → Health check + model info
  POST /predict  → Personality prediction
"""

import os
import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Logging setup 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Personality Predictor API",
    description=(
        "Predicts whether a person is an **Introvert** or **Extrovert** "
        "based on behavioral and social features using a trained Random Forest model."
    ),
    version="1.0.0",
    docs_url="/docs",      
    redoc_url="/redoc",    
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "personality_model.joblib")

try:
    artifact = joblib.load(MODEL_PATH)
    model          = artifact["model"]
    feature_cols   = artifact["feature_columns"]
    label_map      = artifact["label_map"]
    fill_values    = artifact["fill_values"]
    yes_no_cols    = artifact["yes_no_cols"]
    numeric_cols   = artifact["numeric_cols"]
    model_accuracy = artifact.get("model_accuracy", "N/A")
    cv_mean        = artifact.get("cv_mean", "N/A")
    logger.info(f"Model loaded successfully | Accuracy: {model_accuracy}")
except FileNotFoundError:
    logger.error(f"Model file not found at: {MODEL_PATH}")
    raise RuntimeError(
        f"Model artifact not found at '{MODEL_PATH}'. "
        "Please run the training notebook first."
    )


# Pydantic Input Schema 

class PersonalityInput(BaseModel):
    """
    Input schema for the /predict endpoint.
    All fields are optional — missing values will be imputed using training medians/modes.
    """

    Time_spent_Alone: float = Field(
        ...,
        ge=0, le=11,
        description="Average hours spent alone per day (0–11)",
        example=7.0
    )
    Stage_fear: str = Field(
        ...,
        description="Do you have stage fear? ('Yes' or 'No')",
        example="Yes"
    )
    Social_event_attendance: float = Field(
        ...,
        ge=0, le=10,
        description="Number of social events attended per month (0–10)",
        example=2.0
    )
    Going_outside: float = Field(
        ...,
        ge=0, le=10,
        description="How often you go outside per week (0–10)",
        example=2.0
    )
    Drained_after_socializing: str = Field(
        ...,
        description="Do you feel drained after socializing? ('Yes' or 'No')",
        example="Yes"
    )
    Friends_circle_size: float = Field(
        ...,
        ge=0, le=15,
        description="Number of close friends (0–15)",
        example=3.0
    )
    Post_frequency: float = Field(
        ...,
        ge=0, le=10,
        description="How often you post on social media per week (0–10)",
        example=1.0
    )

    # Validators for Yes/No fields
    @field_validator("Stage_fear", "Drained_after_socializing")
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        """Ensure Yes/No fields are valid."""
        if v.strip().lower() not in ("yes", "no"):
            raise ValueError("Must be 'Yes' or 'No'")
        return v.strip().capitalize()  

    class Config:
        json_schema_extra = {
            "example": {
                "Time_spent_Alone": 7.0,
                "Stage_fear": "Yes",
                "Social_event_attendance": 2.0,
                "Going_outside": 2.0,
                "Drained_after_socializing": "Yes",
                "Friends_circle_size": 3.0,
                "Post_frequency": 1.0
            }
        }


# Output Schema
class PredictionOutput(BaseModel):
    prediction: str = Field(..., description="'Introvert' or 'Extrovert'")
    confidence: float = Field(..., description="Model confidence score (0.0–1.0)")


class HealthOutput(BaseModel):
    status: str
    model: str
    accuracy: float
    cv_score: float
    features: list


# Helper: preprocess input
def preprocess_input(data: PersonalityInput) -> pd.DataFrame:
    """
    Convert validated Pydantic input to a DataFrame row
    ready for model prediction.

    This mirrors EXACTLY what the training notebook did:
    1. Map Yes/No → 1/0
    2. Arrange features in correct column order
    """
    row = {}

    for col in feature_cols:
        val = getattr(data, col)
        if col in yes_no_cols:
            # Map Yes → 1.0, No → 0.0
            row[col] = 1.0 if str(val).lower() == "yes" else 0.0
        else:
            row[col] = float(val)

    df_input = pd.DataFrame([row])[feature_cols]
    return df_input


# Routes 

@app.get("/", tags=["General"])
def root():
    """Welcome message and API info."""
    return {
        "message": "Personality Predictor API is running...",
        "docs"   : "/docs",
        "health" : "/health",
        "predict": "POST /predict"
    }


@app.get("/health", response_model=HealthOutput, tags=["General"])
def health_check():
    """
    Health check endpoint.
    Returns model status and performance metrics.
    """
    return HealthOutput(
        status   = "ok",
        model    = "RandomForestClassifier",
        accuracy = model_accuracy,
        cv_score = cv_mean,
        features = feature_cols
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(data: PersonalityInput):
    """
    Predict whether a person is an Introvert or Extrovert.

    **Input**: JSON with 7 behavioral features.
    **Output**: `{ "prediction": "Introvert", "confidence": 0.87 }`
    """
    try:
        df_input = preprocess_input(data)
        logger.info(f"Predicting for input: {df_input.to_dict(orient='records')[0]}")

        prediction_idx = model.predict(df_input)[0]
        probabilities  = model.predict_proba(df_input)[0]

        prediction_label = label_map[int(prediction_idx)]
        confidence       = float(max(probabilities))

        logger.info(f"Prediction: {prediction_label} | Confidence: {confidence:.4f}")

        return PredictionOutput(
            prediction = prediction_label,
            confidence = round(confidence, 4)
        )

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction. Please try again."
        )
