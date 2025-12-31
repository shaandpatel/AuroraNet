"""
Evaluate the trained LSTM Kp forecasting model.

Steps:
1. Load processed validation/test dataset.
2. Load trained model and scaler.
3. Predict Kp values.
4. Compute evaluation metrics: MSE, MAE, RMSE.
5. Plot predictions vs. actuals.
6. Optionally log results to W&B.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import logging
import wandb
import os
import argparse
import json

from src.models.lstm_model import KpLSTM
from src.utils import setup_logging

# ---------------------------
# Logging
# ---------------------------
setup_logging()
logger = logging.getLogger(__name__)

def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")

    run = wandb.init(project=args.project_name, job_type="evaluation")

    # --- Get Model and Config from W&B Artifact ---
    logger.info(f"Fetching model artifact: {args.model_artifact}")
    artifact = run.use_artifact(args.model_artifact, type='model')
    artifact_dir = artifact.download()

    # Load config from the artifact, overriding command-line args
    config = artifact.metadata
    model_path = os.path.join(artifact_dir, "kp_lstm.pth")

    # Update config with runtime args
    config['data_path'] = args.data_path
    config['batch_size'] = args.batch_size

    # ---------------------------
    # Load Dataset
    # ---------------------------
    logger.info(f"Loading dataset for evaluation from {config['data_path']}...")
    try:
        data = np.load(config['data_path'])
        X, y = data["X"], data["y"]
    except FileNotFoundError:
        logger.error(f"Data file not found at {config['data_path']}. Please generate it first.")
        return

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    # ---------------------------
    # Load Model
    # ---------------------------
    logger.info(f"Loading trained model from {model_path}...")
    input_size = X.shape[2]
    output_size = y.shape[1] if len(y.shape) > 1 else 1
    
    model = KpLSTM(
        input_size=input_size, 
        hidden_size=config['hidden_size'], 
        num_layers=config['num_layers'], 
        output_size=output_size,
        dropout=config['dropout']
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}. Please train a model first.")
        return
    except RuntimeError as e:
        logger.error(f"Error loading model state_dict: {e}")
        logger.error("This might be due to a mismatch in model architecture (e.g., hidden_size, num_layers).")
        return
        
    model.to(DEVICE)
    model.eval()

    wandb.watch(model, log="all", log_freq=10)

    # ---------------------------
    # Prediction and Metrics
    # ---------------------------
    preds, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            preds.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Compute metrics
    mse = np.mean((preds - actuals) ** 2)
    mae = np.mean(np.abs(preds - actuals))
    rmse = np.sqrt(mse)

    logger.info(f"Evaluation Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # Save metrics locally
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = {"test_mse": float(mse), "test_mae": float(mae), "test_rmse": float(rmse)}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")

    wandb.log(metrics)
    
    # 1. Time Series Plot (First 500 points)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actuals[:500], label="Actual Kp")
    ax.plot(preds[:500], label="Predicted Kp")
    ax.set_title("Kp Forecast: Predictions vs Actuals (first 500 points)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Kp index")
    ax.legend()
    
    ts_plot_path = os.path.join(args.output_dir, "time_series_plot.png")
    plt.savefig(ts_plot_path)
    wandb.log({"predictions_vs_actuals_timeseries": wandb.Image(ts_plot_path)})
    plt.close(fig)

    # 2. Scatter Plot (Actual vs Predicted)
    # Flatten arrays to handle potential multi-step outputs uniformly
    flat_actuals = actuals.flatten()
    flat_preds = preds.flatten()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(flat_actuals, flat_preds, alpha=0.3, s=10)
    
    # Draw identity line
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Actual Kp')
    ax.set_ylabel('Predicted Kp')
    ax.set_title('Actual vs Predicted Kp')
    
    scatter_plot_path = os.path.join(args.output_dir, "scatter_plot.png")
    plt.savefig(scatter_plot_path)
    wandb.log({"predictions_vs_actuals_scatter": wandb.Image(scatter_plot_path)})
    plt.close(fig)

    # 3. Residuals Histogram
    residuals = flat_actuals - flat_preds
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title("Distribution of Residuals (Actual - Predicted)")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    
    res_plot_path = os.path.join(args.output_dir, "residuals_histogram.png")
    plt.savefig(res_plot_path)
    wandb.log({"residuals_histogram": wandb.Image(res_plot_path)})
    plt.close(fig)
    
    wandb.finish()

    logger.info("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Kp index forecasting LSTM model.")
    
    # W&B artifact to evaluate
    parser.add_argument("--model_artifact", type=str, default="kp-lstm-model:latest", help="W&B artifact to evaluate. Defaults to the 'latest' alias.")
    
    # Runtime settings
    parser.add_argument("--data_path", type=str, default="datafiles/processed/test_data.npz", help="Path to the test data file.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--project_name", type=str, default="aurora-forecast", help="W&B project name.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save evaluation results (metrics and plots).")

    args = parser.parse_args()
    main(args)
