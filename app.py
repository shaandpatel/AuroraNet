from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
import pandas as pd
import logging
import os
from contextlib import asynccontextmanager

from src.models.infer import load_model_from_artifact, predict_kp
from src.data import fetch_realtime_solarwind, fetch_recent_kp

# Configuration via Environment Variables
MODEL_ARTIFACT = os.getenv("MODEL_ARTIFACT", "aurora-forecast/kp-lstm-model:latest")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESOLUTION = int(os.getenv("RESOLUTION", 24))

# Global variables to hold model state
model_context = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model and scaler when the server starts.
    This prevents reloading 100MB+ files on every single API call.
    """
    logging.info("Server starting up... loading model.")
    model, scaler, config = load_model_from_artifact(MODEL_ARTIFACT, DEVICE)
    
    if not model:
        logging.error("Failed to load model from W&B.")
        # In production, you might want to raise an error here to fail the deployment
    
    model_context["model"] = model
    model_context["scaler"] = scaler
    model_context["config"] = config
    yield
    # Clean up resources if needed
    model_context.clear()

app = FastAPI(title="Aurora Kp Forecast API", lifespan=lifespan)

class PredictionResponse(BaseModel):
    forecast_timestamps: List[str]
    kp_predictions: List[float]
    source: str

@app.get("/health")
def health_check():
    return {"status": "active", "device": DEVICE, "model_artifact": MODEL_ARTIFACT}

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint():
    """
    Triggers a real-time inference using NOAA data.
    """
    model = model_context.get("model")
    scaler = model_context.get("scaler")
    config = model_context.get("config")

    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # 1. Fetch Data (Reusing your logic from infer.py)
        sw_df = fetch_realtime_solarwind()
        kp_df = fetch_recent_kp(hours=168)
        
        sw_df = sw_df.sort_values('time_tag')
        kp_df = kp_df.sort_values('time_tag')
        
        input_df = pd.merge_asof(sw_df, kp_df, on='time_tag', direction='backward')
        
        if 'kp_index' in input_df.columns:
            input_df['kp_index'] = input_df['kp_index'].fillna(method='bfill')

        # 2. Run Inference
        seq_length = config.get('seq_length', 60) # Default fallback
        kp_pred = predict_kp(input_df, model, scaler, seq_length, RESOLUTION, DEVICE)

        if kp_pred is None:
            raise HTTPException(status_code=400, detail="Prediction returned None (insufficient data?).")

        # 3. Format Response
        last_time = pd.to_datetime(input_df['time_tag'].iloc[-1])
        timestamps = []
        for i in range(len(kp_pred)):
            ts = last_time + pd.Timedelta(minutes=RESOLUTION * (i + 1))
            timestamps.append(ts.strftime('%Y-%m-%d %H:%M'))

        return {
            "forecast_timestamps": timestamps,
            "kp_predictions": kp_pred.tolist(),
            "source": "NOAA Real-time Fetch"
        }

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))