from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import pandas as pd
import logging
import os
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# Import your modules
from src.models.infer import load_model_from_artifact, predict_kp
from src.data import fetch_realtime_solarwind, fetch_recent_kp

# --- CONFIGURATION ---
MODEL_ARTIFACT = os.getenv("MODEL_ARTIFACT", "aurora-forecast/kp-lstm-model:latest")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESOLUTION = int(os.getenv("RESOLUTION", 60))
CACHE_DURATION_MINUTES = 15 

# --- GLOBAL STATE & CACHE ---
model_context = {}
data_cache: Dict[str, Any] = {
    "data": None,
    "last_updated": None,
    "source": None
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once on startup."""
    logger.info("Server starting... loading model artifact.")
    try:
        model, scaler, config = load_model_from_artifact(MODEL_ARTIFACT, DEVICE)
        if not model:
            raise ValueError("Model failed to load.")
        
        model_context["model"] = model
        model_context["scaler"] = scaler
        model_context["config"] = config
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load model: {e}")
        # In production, we might want to crash here, but for now we log it.
    
    yield
    model_context.clear()

app = FastAPI(title="Aurora Kp Forecast API", lifespan=lifespan)

class PredictionResponse(BaseModel):
    forecast_timestamps: List[str]
    kp_predictions: List[float]
    source: str
    cached_at: Optional[str] = None
    warning: Optional[str] = None

@app.get("/health")
def health_check():
    return {"status": "active", "device": DEVICE}

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint():
    """
    Returns Kp forecast. Uses in-memory caching to reduce load on NOAA.
    """
    model = model_context.get("model")
    scaler = model_context.get("scaler")
    config = model_context.get("config")

    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")

    # --- CACHE CHECK ---
    now = datetime.now()
    is_cache_valid = False
    
    if data_cache["data"] is not None and data_cache["last_updated"] is not None:
        age = now - data_cache["last_updated"]
        if age < timedelta(minutes=CACHE_DURATION_MINUTES):
            is_cache_valid = True
            logger.info("Serving valid cached data.")

    # --- FETCH NEW DATA IF NEEDED ---
    # We enter this block if cache is empty OR expired
    if not is_cache_valid:
        try:
            logger.info("Cache expired or empty. Fetching fresh NOAA data...")
            sw_df = fetch_realtime_solarwind()
            kp_df = fetch_recent_kp(hours=168)

            # Basic validation
            if sw_df.empty or kp_df.empty:
                raise ValueError("Fetched datasets are empty.")

            sw_df = sw_df.sort_values('time_tag')
            kp_df = kp_df.sort_values('time_tag')
            
            # Merge logic
            input_df = pd.merge_asof(sw_df, kp_df, on='time_tag', direction='backward')
            if 'kp_index' in input_df.columns:
                input_df['kp_index'] = input_df['kp_index'].bfill()

            # Update Cache
            data_cache["data"] = input_df
            data_cache["last_updated"] = now
            data_cache["source"] = "NOAA Real-time"
            logger.info("Cache successfully updated.")

        except Exception as e:
            logger.error(f"Data Fetch Failed: {e}")
            # --- FALLBACK LOGIC ---
            # If fetch fails, check if we have any old cache (stale data)
            if data_cache["data"] is not None:
                logger.warning("Serving STALE data due to fetch failure.")
                # We do not update the timestamp, so we try fetching again next time
            else:
                # No data at all? We must fail.
                raise HTTPException(
                    status_code=502, 
                    detail=f"External NOAA data unavailable and cache is empty. Error: {str(e)}"
                )

    # --- INFERENCE ---
    try:
        input_df = data_cache["data"]
        seq_length = config.get('seq_length', 60)
        
        kp_pred = predict_kp(input_df, model, scaler, seq_length, RESOLUTION, DEVICE)

        if kp_pred is None:
            raise ValueError("Model inference returned None.")

        # Format timestamps
        last_time = pd.to_datetime(input_df['time_tag'].iloc[-1])
        timestamps = []
        for i in range(len(kp_pred)):
            ts = last_time + timedelta(minutes=RESOLUTION * (i + 1))
            timestamps.append(ts.strftime('%Y-%m-%d %H:%M'))

        # Prepare response info
        response_source = data_cache["source"]
        warning_msg = None
        
        # If the data is old (stale fallback), add a warning
        age = datetime.now() - data_cache["last_updated"]
        if age > timedelta(minutes=CACHE_DURATION_MINUTES + 5):
            response_source = "Stale Cache"
            warning_msg = f"Data is {int(age.total_seconds() // 60)} mins old. Live updates unavailable."

        return {
            "forecast_timestamps": timestamps,
            "kp_predictions": kp_pred.tolist(),
            "source": response_source,
            "cached_at": data_cache["last_updated"].strftime('%H:%M:%S UTC'),
            "warning": warning_msg
        }

    except Exception as e:
        logger.error(f"Inference/Formatting Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Calculation Error.")