"""
Train an LSTM model to forecast Kp index using processed solar-wind features.

Features:
- Loads preprocessed dataset
- Splits into train/validation sets
- Trains LSTM
- Logs metrics and hyperparameters to Weights & Biases
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
    # Load the entire dataset into RAM. This is the fastest method if you have enough memory.
    with np.load(config.data_path) as data:
        X_full = data["X"]
        y_full = data["y"]

    # Split training data into Train and Validation (e.g., 80/20)
    # We do this internally so the Test set (test_data.npz) remains completely unseen.
    split_idx = int(len(X_full) * 0.8)
    X_train, X_val = X_full[:split_idx], X_full[split_idx:]
    y_train, y_val = y_full[:split_idx], y_full[split_idx:]

    # Convert to PyTorch Tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val)

    # Use the standard, highly optimized TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    num_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
    
    logger.info(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # ---------------------------
    # Initialize model
    # ---------------------------
    input_size = X_train_tensor.shape[2]
    output_size = y_train_tensor.shape[1] if len(y_train_tensor.shape) > 1 else 1


    # --- Add sequence length to config for inference ---
    # This ensures the inference script knows the model's expected input window size.
    seq_length = X_train_tensor.shape[1]
    wandb.config.update({
        "seq_length": seq_length,
        "input_size": input_size,
        "output_size": output_size
    }, allow_val_change=True)

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

    # Parameter calculation
    # total_params = sum(p.numel() for p in model.parameters())
    # print(total_params)

    # ---------------------------
    # Training loop
    # ---------------------------
    logger.info("Starting training...")
    best_val_loss = float('inf')
    early_stopping_counter = 0

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
            early_stopping_counter = 0  # Reset counter on improvement
        else:
            early_stopping_counter += 1
            logger.info(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{config.early_stopping_patience}")

        # Check for early stopping
        if early_stopping_counter >= config.early_stopping_patience:
            logger.info(f"Stopping early as validation loss has not improved for {config.early_stopping_patience} epochs.")
            break


    # ---------------------------
    # Finalize and Save Artifact
    # ---------------------------
    logger.info(f"Training finished. Best model saved at {args.model_path} with validation loss {best_val_loss:.4f}")

    # Save model as a wandb artifact
    artifact = wandb.Artifact("kp-lstm-model", type='model', metadata=dict(config))
    
    # Add model file
    artifact.add_file(args.model_path)
    # Add scaler file (assuming it's in the same dir as the data)
    if os.path.exists(args.scaler_path):
        artifact.add_file(args.scaler_path)

    logger.info(f"Logging artifact '{artifact.name}' to W&B with alias 'latest'...")
    # Log the artifact with an alias that always points to the latest version.
    wandb.log_artifact(artifact, aliases=["latest"])

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Kp index forecasting LSTM model.")
    parser.add_argument("--data_path", type=str, default="datafiles/processed/training_data.npz", help="Path to the training data file.")
    parser.add_argument("--model_path", type=str, default="models/kp_lstm.pth", help="Path to save the trained model.")
    parser.add_argument("--scaler_path", type=str, default="models/scaler.pkl", help="Path to the fitted scaler file.")
    parser.add_argument("--project_name", type=str, default="aurora-forecast", help="W&B project name.")
    parser.add_argument("--seed", type=int, default=27, help="Random seed for reproducibility.")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=.00633, help="Learning rate.")
    parser.add_argument("--hidden_size", type=int, default=64, help="LSTM hidden size.")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.2314, help="Dropout rate.")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of epochs to wait for validation loss improvement before stopping.")

    args = parser.parse_args()
    main(args)
