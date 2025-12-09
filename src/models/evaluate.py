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

from src.models.lstm_model import KpLSTM
from src.utils import setup_logging

# ---------------------------
# Logging
# ---------------------------
setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------
# Config
# ---------------------------
DATA_PATH = "data/processed/training_dataset.npz"
MODEL_PATH = "models/saved_model.pth"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_WANDB = True  # set False if you don't want W&B logging

# ---------------------------
# Load Dataset
# ---------------------------
logger.info("Loading dataset for evaluation...")
data = np.load(DATA_PATH)
X, y = data["X"], data["y"]

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# Load Model
# ---------------------------
logger.info("Loading trained model...")
input_size = X.shape[2]
output_size = y.shape[1] if len(y.shape) > 1 else 1
model = KpLSTM(input_size=input_size, hidden_size=64, num_layers=2, output_size=output_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------------------
# W&B initialization
# ---------------------------
if USE_WANDB:
    wandb.init(project="aurora-eval", config={"batch_size": BATCH_SIZE})
    wandb.watch(model, log="all", log_freq=10)

# ---------------------------
# Prediction and Metrics
# ---------------------------
preds, actuals = [], []

criterion = nn.MSELoss()
total_loss = 0.0

with torch.no_grad():
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        preds.append(outputs.cpu().numpy())
        actuals.append(y_batch.cpu().numpy())

preds = np.concatenate(preds, axis=0)
actuals = np.concatenate(actuals, axis=0)

# Compute metrics
mse = np.mean((preds - actuals) ** 2)
mae = np.mean(np.abs(preds - actuals))
rmse = np.sqrt(mse)

logger.info(f"Evaluation Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

if USE_WANDB:
    wandb.log({"MSE": mse, "MAE": mae, "RMSE": rmse})
    wandb.finish()

# ---------------------------
# Plot Predictions vs Actuals
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(actuals[:500], label="Actual Kp")
plt.plot(preds[:500], label="Predicted Kp")
plt.title("Kp Forecast: Predictions vs Actuals (first 500 points)")
plt.xlabel("Time step")
plt.ylabel("Kp index")
plt.legend()
plt.tight_layout()
plt.show()
