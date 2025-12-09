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

    # For reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")

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

    # --- Add sequence length to config for inference ---
    # This ensures the inference script knows the model's expected input window size.
    seq_length = X.shape[1]
    wandb.config.update({"seq_length": seq_length}, allow_val_change=True)

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
    best_val_loss = float('inf')

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

        # Save the best model based on validation loss
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            model_dir = os.path.dirname(args.model_path)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), args.model_path)
            logger.info(f"New best model saved to {args.model_path} (Val Loss: {best_val_loss:.4f})")


    # ---------------------------
    # Save model
    # ---------------------------
    logger.info(f"Training finished. Best model saved at {args.model_path} with validation loss {best_val_loss:.4f}")

    # Save model as a wandb artifact
    # Use a consistent name for the artifact so we can easily reference it.
    artifact = wandb.Artifact("kp-lstm-model", type='model', metadata=dict(config))
    
    # Add model file
    artifact.add_file(args.model_path)
    # Add scaler file (assuming it's in the same dir as the data)
    if os.path.exists(args.scaler_path):
        artifact.add_file(args.scaler_path)

    # Log the artifact with an alias that always points to the latest version.
    wandb.log_artifact(artifact, aliases=["latest"])

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Kp index forecasting LSTM model.")
    parser.add_argument("--data_path", type=str, default="datafiles/processed/training_data.npz", help="Path to the training data file.")
    parser.add_argument("--model_path", type=str, default="models/kp_lstm.pth", help="Path to save the trained model.")
    parser.add_argument("--scaler_path", type=str, default="models/scaler.pkl", help="Path to the fitted scaler file.")
    parser.add_argument("--project_name", type=str, default="aurora-forecast", help="W&B project name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
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
