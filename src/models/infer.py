"""
Inference pipeline for Kp forecasting.

Steps:
1. Load trained model and scaler.
2. Accept raw/current solar wind features (or preprocessed data).
3. Apply preprocessing, scaling, and feature engineering.
4. Predict Kp index (multi-step horizon).
5. Return predictions in a user-friendly format.
"""

import torch
import numpy as np
import pandas as pd
import logging
from src.models.lstm_model import KpLSTM
from src.data import clean_solarwind, add_time_features, add_moving_averages, fetch_realtime_solarwind, fetch_recent_kp
from src.utils import setup_logging
import argparse
import joblib
import wandb
import os



# ---------------------------
# Logging
# ---------------------------
setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------
# Load model & scaler
# ---------------------------
def load_model_from_artifact(model_artifact, device):
    # 1. Try Loading Locally First
    # Define candidate paths for robustness (CWD vs Absolute vs Env Var)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    candidates = [
        # 1. Check if the artifact argument is actually a file path
        (model_artifact, os.path.join(os.path.dirname(model_artifact) if model_artifact else "", "scaler.pkl")),
        # 2. Check default relative path (depends on CWD)
        ("models/kp_lstm.pth", "models/scaler.pkl"),
        # 3. Check absolute path relative to project root (robust for Docker)
        (os.path.join(base_dir, "models", "kp_lstm.pth"), os.path.join(base_dir, "models", "scaler.pkl"))
    ]

    model_path = None
    scaler_path = None

    for mp, sp in candidates:
        if mp and os.path.exists(mp) and os.path.exists(sp):
            model_path = mp
            scaler_path = sp
            break

    if model_path:
        logger.info(f"Loading model from local path: {os.path.abspath(model_path)}")
        scaler = joblib.load(scaler_path)
        
        config = {
            'input_size': scaler.n_features_in_ + 8, 
            'hidden_size': 64,
            'num_layers': 1,
            'output_size': 6,
            'dropout': 0.2314,
            'seq_length': 72
        }
        
        model = KpLSTM(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['num_layers'], output_size=config['output_size'], dropout=config['dropout'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, scaler, config

    # 2. Fallback to W&B (Only works if user has credentials)
    logger.info(f"Local artifacts not found. Attempting W&B download...")
    logger.warning(f"Checked paths: {[c[0] for c in candidates]}")
    logger.info(f"Fetching model artifact: {model_artifact}")
    run = wandb.init(project="aurora-forecast", job_type="inference")
    artifact = run.use_artifact(model_artifact, type='model')
    artifact_dir = artifact.download()
    run.finish()

    config = artifact.metadata
    model_path = os.path.join(artifact_dir, "kp_lstm.pth")
    scaler_path = os.path.join(artifact_dir, "scaler.pkl")

    logger.info(f"Loading scaler from {scaler_path}")
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        logger.error(f"scaler.pkl not found in the artifact. Make sure it was saved during training.")
        return None, None, None

    logger.info(f"Loading model from {model_path}")
    
    # Determine input_size
    if 'input_size' in config:
        input_size = config['input_size']
    else:
        # Fallback for older artifacts: Base features + (4 columns * 2 windows)
        input_size = scaler.n_features_in_ + 8
        logger.warning(f"'input_size' missing in config. Inferred {input_size} (scaler features + 8 MA features).")

    model = KpLSTM(input_size=input_size, hidden_size=config['hidden_size'], num_layers=config['num_layers'], output_size=config['output_size'], dropout=config['dropout'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, scaler, config

# ---------------------------
# Prediction function
# ---------------------------
def predict_kp(solarwind_df, model, scaler, seq_length, resolution, device):
    """
    Given a DataFrame of solar wind features, predict next Kp index(es).
    
    Args:
        solarwind_df (DataFrame): raw solar wind data
        model (KpLSTM): trained PyTorch model
        scaler (sklearn scaler): fitted scaler from training
        seq_length (int): number of timesteps in the input window
        resolution (int): data resolution in minutes (to resample input)
        device (str): 'cpu' or 'cuda'

    Returns:
        np.array: predicted Kp index values
    """
    # 0. Resample to match model resolution
    # We do this before cleaning to ensure the time steps match the training data cadence.
    df = solarwind_df.copy()
    if resolution and resolution > 0:
        if 'time_tag' in df.columns:
            df = df.set_index('time_tag')
        
        # Resample to the target resolution (e.g., 24min) using mean aggregation
        df = df.resample(f'{resolution}min').mean()
        df = df.reset_index()
        # Note: Resampling might create NaNs for missing periods; clean_solarwind handles this.

    # 1. Preprocessing Pipeline (Replicating build_dataset.py)
    # Clean
    df = clean_solarwind(df)
    
    # Time features
    df = add_time_features(df)
    
    # --- Add Lagged Kp Feature ---
    # Replicate the lag logic from build_dataset.py
    if 'kp_index' in df.columns:
        # Shift by 1. Use ffill to handle gaps in stream, then bfill for the start.
        # This ensures we always have a valid 'kp_prev' even if real-time data is spotty.
        df['kp_prev'] = df['kp_index'].shift(1).fillna(method='ffill').fillna(method='bfill')
    elif 'kp_prev' in scaler.feature_names_in_:
        logger.warning("Model expects 'kp_prev' feature, but 'kp_index' is missing from input data.")

    # Scale
    try:
        feature_cols = scaler.feature_names_in_
        # Check if required columns exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
             logger.error(f"Input data missing columns required for scaling: {missing}")
             return None
        
        # Scale features (keep as DataFrame to add MAs)
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])
        
    except Exception as e:
        logger.error(f"Preprocessing/Scaling failed: {e}")
        return None

    # Moving Averages (Hardcoded to match build_dataset.py)
    ma_cols = ['speed', 'density', 'bz', 'b']
    
    # Check if we have enough history for stable moving averages
    max_window = 50
    if len(df_scaled) < max_window:
        logger.warning(f"Input data length ({len(df_scaled)}) is shorter than MA window ({max_window}). Features will be less accurate.")

    df_final = add_moving_averages(df_scaled, columns=ma_cols, window=10)
    df_final = add_moving_averages(df_final, columns=ma_cols, window=50)
    
    # Construct final feature list in the correct order
    # Order: [scaler_features, ma10_features, ma50_features]
    final_cols = list(feature_cols)
    final_cols += [f"{c}_ma10" for c in ma_cols]
    final_cols += [f"{c}_ma50" for c in ma_cols]
    
    # Check if final columns exist
    missing_final = [c for c in final_cols if c not in df_final.columns]
    if missing_final:
        logger.error(f"Missing final engineered features: {missing_final}")
        return None
        
    df_ordered = df_final[final_cols]

    # 2. Check sequence length
    if len(df_ordered) < seq_length:
        logger.error(f"Not enough data to create a sequence. Need {seq_length} rows, but got {len(df_ordered)}.")
        return None
    
    # 3. Take the last `seq_length` rows
    input_sequence = df_ordered.tail(seq_length).values
    
    # 4. Convert to tensor and add a batch dimension
    X_seq_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 5. Predict
    with torch.no_grad():
        pred = model(X_seq_tensor)
    return pred.cpu().numpy().flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Kp index forecasting.")
    parser.add_argument("--model_artifact", type=str, default='kp-lstm-model:latest', help="W&B artifact to use for inference, e.g., 'aurora-forecast/kp-lstm-model-abc123:v0'")
    parser.add_argument("--input_data_path", type=str, default=None, help="Optional path to local input solar wind data (CSV). If not provided, fetches real-time data.")
    parser.add_argument("--resolution", type=int, default=24, help="Data resolution in minutes (default: 24). Should match the training resolution.")
    
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model, scaler, and config from the artifact
    model, scaler, config = load_model_from_artifact(args.model_artifact, DEVICE)

    if model and scaler and config:
        input_df = None
        
        # Default to real-time unless input_data_path is explicitly provided
        if args.input_data_path is None:
            logger.info("No local input file provided. Fetching real-time solar wind data from NOAA...")
            try:
                # Fetch both Solar Wind and Kp data
                sw_df = fetch_realtime_solarwind()
                kp_df = fetch_recent_kp(hours=168)  # Fetch 7 days to ensure full overlap with SW data
                
                # Merge Kp onto Solar Wind data (SW is higher resolution)
                # We use merge_asof with direction='backward' to assign the most recent known Kp 
                # to each solar wind timestamp.
                sw_df = sw_df.sort_values('time_tag')
                kp_df = kp_df.sort_values('time_tag')
                
                input_df = pd.merge_asof(sw_df, kp_df, on='time_tag', direction='backward')
                
                # Fill any initial gaps if SW data starts slightly before Kp data
                if 'kp_index' in input_df.columns:
                    input_df['kp_index'] = input_df['kp_index'].fillna(method='bfill')

                logger.info(f"Fetched {len(input_df)} rows of real-time data (Solar Wind + Kp).")
            except Exception as e:
                logger.error(f"Failed to fetch real-time data: {e}")
        else:
            logger.info(f"Loading data from local file: {args.input_data_path}")
            try:
                input_df = pd.read_csv(args.input_data_path, index_col=0, parse_dates=True)
                # Ensure time_tag is a column (not index) for clean_solarwind compatibility
                if 'time_tag' not in input_df.columns:
                    input_df = input_df.reset_index()
                    if 'time_tag' not in input_df.columns:
                        # Fallback: assume first column is time if not named 'time_tag'
                        input_df.rename(columns={input_df.columns[0]: 'time_tag'}, inplace=True)
            except FileNotFoundError:
                logger.error(f"Input data file not found at {args.input_data_path}")

        if input_df is not None:
            # Get the sequence length from the training config
            seq_length = config.get('seq_length')
            
            if seq_length:
                kp_pred = predict_kp(input_df, model, scaler, seq_length, args.resolution, DEVICE)
                if kp_pred is not None:
                    logger.info(f"Predicted Kp: {kp_pred[0]:.4f}")
                    
                    # Use the explicit resolution for forecasting timestamps
                    last_time = pd.to_datetime(input_df['time_tag'].iloc[-1])
                    logger.info(f"Forecast starting from {last_time} (Resolution: {args.resolution} min):")
                    
                    for i, val in enumerate(kp_pred):
                        forecast_time = last_time + pd.Timedelta(minutes=args.resolution * (i + 1))
                        logger.info(f"  T+{args.resolution * (i + 1)}m ({forecast_time.strftime('%Y-%m-%d %H:%M')}): Kp {val:.2f}")
            else:
                logger.error("'seq_length' not found in model artifact metadata. Cannot run inference.")
    else:
        logger.error("Failed to load model from artifact. Aborting inference.")
    
