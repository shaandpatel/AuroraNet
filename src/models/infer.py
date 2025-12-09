"""
Inference pipeline for Kp forecasting.

Steps:
1. Load trained model and scaler.
2. Accept raw/current solar wind features (or preprocessed data).
3. Apply preprocessing, scaling, and feature engineering.
4. Predict Kp index (single or multi-step horizon).
5. Return predictions in a user-friendly format.
"""

import torch
import numpy as np
import pandas as pd
import logging
from src.models.lstm_model import KpLSTM
from src.data import clean_solarwind, add_time_features, make_supervised
# `add_features` is not a valid import, assuming it was meant to be the combination of other feature engineering steps.
from src.utils import setup_logging

# ---------------------------
# Logging
# ---------------------------
setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------
# Configuration
# ---------------------------
MODEL_PATH = "models/saved_model.pth"
SCALER_PATH = "models/scaler.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Example input sequence length (should match training window)
SEQ_LENGTH = 24

# ---------------------------
# Load model & scaler
# ---------------------------
def load_model(model_path=MODEL_PATH, input_size=5, output_size=1, hidden_size=64, num_layers=2):
    """
    Load trained LSTM model for inference.
    """
    model = KpLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ---------------------------
# Prediction function
# ---------------------------
def predict_kp(solarwind_df, model, scaler, seq_length=SEQ_LENGTH):
    """
    Given a DataFrame of solar wind features, predict next Kp index(es).
    
    Args:
        solarwind_df (DataFrame): raw or cleaned solar wind data
        model (KpLSTM): trained PyTorch model
        scaler (sklearn scaler): fitted scaler from training
        seq_length (int): number of timesteps in input window

    Returns:
        np.array: predicted Kp index values
    """
    # 1. Clean data
    df_clean = clean_solarwind(solarwind_df)

    # 2. Feature engineering (must match build_dataset.py)
    df_feat = add_time_features(df_clean)

    # Ensure columns are in the same order as during training
    feature_cols = [col for col in df_feat.columns if col in scaler.feature_names_in_]
    df_feat = df_feat[feature_cols]

    # 3. Scaling
    df_scaled = pd.DataFrame(scaler.transform(df_feat), columns=feature_cols, index=df_feat.index)
    df_scaled['time_tag'] = df_feat.index # make_supervised expects time_tag

    # 4. Make supervised input sequence
    X_seq, _ = make_supervised(df_scaled, feature_cols, target_col='kp_index', window=seq_length, horizon=1, for_inference=True)
    X_seq_tensor = torch.tensor(X_seq[-1:], dtype=torch.float32).to(DEVICE)  # take last sequence

    # 5. Predict
    with torch.no_grad():
        pred = model(X_seq_tensor)
    return pred.cpu().numpy().flatten()

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    import joblib

    logger.info("Loading scaler and model...")
    scaler = joblib.load(SCALER_PATH)
    model = load_model(input_size=scaler.scale_.shape[0])

    # Example: load last 48 hours of solar wind data
    example_data = pd.read_csv("data/examples/solarwind_example.csv", index_col=0)
    
    kp_pred = predict_kp(example_data, model, scaler)
    logger.info(f"Predicted Kp: {kp_pred}")
