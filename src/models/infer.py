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
    """
    Download a model artifact from W&B and load the model and scaler.
    """
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
    input_size = scaler.n_features_in_
    model = KpLSTM(input_size=input_size, hidden_size=config['hidden_size'], num_layers=config['num_layers'], output_size=config['output_size'], dropout=config['dropout'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, scaler, config

# ---------------------------
# Prediction function
# ---------------------------
def predict_kp(solarwind_df, model, scaler, seq_length, device):
    """
    Given a DataFrame of solar wind features, predict next Kp index(es).
    
    Args:
        solarwind_df (DataFrame): raw or cleaned solar wind data
        model (KpLSTM): trained PyTorch model
        scaler (sklearn scaler): fitted scaler from training
        seq_length (int): number of timesteps in the input window
        device (str): 'cpu' or 'cuda'

    Returns:
        np.array: predicted Kp index values
    """
    # The input `solarwind_df` should already be cleaned and feature-engineered,
    # containing all the columns the model was trained on.
    
    # 1. Ensure columns are in the same order as during training
    try:
        feature_cols = scaler.feature_names_in_
        df_ordered = solarwind_df[feature_cols]
    except KeyError as e:
        logger.error(f"Input data is missing required columns: {e}")
        return None
    
    # 2. Check if we have enough data for a full sequence
    if len(df_ordered) < seq_length:
        logger.error(f"Not enough data to create a sequence. Need {seq_length} rows, but got {len(df_ordered)}.")
        return None
    
    # 3. Take the last `seq_length` rows and scale them
    input_sequence = df_ordered.tail(seq_length)
    input_scaled = scaler.transform(input_sequence)
    
    # 4. Convert to tensor and add a batch dimension
    X_seq_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    if X_seq_tensor.shape[1] != seq_length:
        logger.error(f"Not enough data to create a sequence of length {seq_length}. Need at least {seq_length} rows after cleaning.")
        return None
    
    # 5. Predict
    with torch.no_grad():
        pred = model(X_seq_tensor)
    return pred.cpu().numpy().flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Kp index forecasting.")
    parser.add_argument("--model_artifact", type=str, required=True, help="W&B artifact to use for inference, e.g., 'aurora-forecast/kp-lstm-model-abc123:v0'")
    parser.add_argument("--input_data_path", type=str, default="datafiles/examples/solarwind_example.csv", help="Path to the input solar wind data (CSV).")
    
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model, scaler, and config from the artifact
    model, scaler, config = load_model_from_artifact(args.model_artifact, DEVICE)

    if model and scaler and config:
        try:
            input_df = pd.read_csv(args.input_data_path, index_col=0, parse_dates=True)
        except FileNotFoundError:
            logger.error(f"Input data file not found at {args.input_data_path}")
        else:
            # Get the sequence length from the training config
            seq_length = config.get('seq_length')
            
            if seq_length:
                kp_pred = predict_kp(input_df, model, scaler, seq_length, DEVICE)
                if kp_pred is not None:
                    logger.info(f"Predicted Kp: {kp_pred[0]:.4f}")
            else:
                logger.error("'seq_length' not found in model artifact metadata. Cannot run inference.")
    else:
        logger.error("Failed to load model from artifact. Aborting inference.")
    
