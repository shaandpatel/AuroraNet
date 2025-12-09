"""
Train an LSTM model to forecast Kp index using processed solar-wind features.

Features:
- Loads preprocessed dataset
- Splits into train/validation sets
- Trains LSTM
- Logs metrics and hyperparameters to Weights & Biases (wandb)
- Saves trained model and scaler
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import logging
import wandb
import os
import argparse

from src.utils import setup_logging
from src.models.lstm_model import KpLSTM

# ---------------------------
# Logging
# ---------------------------
setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------
def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")

    # ---------------------------
    # Initialize W&B
    # ---------------------------
    wandb.init(project=args.project_name, config=args)
    config = wandb.config

    # ---------------------------
    # Load Dataset
    # ---------------------------
    logger.info(f"Loading dataset from {config.data_path}...")
    data = np.load(config.data_path)
    X, y = data["X"], data["y"]

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    # --- Time-series split (NO random shuffling) ---
    val_size = int(config.val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    # Create indices for sequential split
    indices = list(range(len(dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    logger.info(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # ---------------------------
    # Initialize model
    # ---------------------------
    input_size = X.shape[2]
    output_size = y.shape[1] if len(y.shape) > 1 else 1

    model = KpLSTM(
        input_size=input_size, 
        hidden_size=config.hidden_size, 
        num_layers=config.num_layers, 
        output_size=output_size,
        dropout=config.dropout
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    wandb.watch(model, log="all", log_freq=100)

    # ---------------------------
    # Training loop
    # ---------------------------
    logger.info("Starting training...")
    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        train_loss_mean = np.mean(train_losses)
        val_loss_mean = np.mean(val_losses)
        logger.info(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss_mean:.4f}, Val Loss: {val_loss_mean:.4f}")

        # Log to W&B
        wandb.log({"epoch": epoch+1, "train_loss": train_loss_mean, "val_loss": val_loss_mean})

    # ---------------------------
    # Save model
    # ---------------------------
    model_dir = os.path.dirname(args.model_path)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    logger.info(f"Saved trained model to {args.model_path}")

    # Save model as a wandb artifact
    artifact = wandb.Artifact('kp-lstm-model', type='model')
    artifact.add_file(args.model_path)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Kp index forecasting LSTM model.")
    parser.add_argument("--data_path", type=str, default="data/processed/training_data.npz", help="Path to the training data file.")
    parser.add_argument("--model_path", type=str, default="models/kp_lstm.pth", help="Path to save the trained model.")
    parser.add_argument("--project_name", type=str, default="aurora-forecast", help="W&B project name.")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden_size", type=int, default=64, help="LSTM hidden size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Proportion of data to use for validation.")

    args = parser.parse_args()
    main(args)
