"""
Shared preprocessing tools for training and inference:
- Cleaning solar wind
- Scaling features
- Sliding windows for LSTM/Transformer
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clean_solarwind(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize solar-wind data for model input.

    - Sorts by time
    - Fills missing values
    - Clips extreme outliers for numerical stability

    Args:
        df (pd.DataFrame): Raw solar-wind DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for scaling or merging.
    """

    df = df.copy()
    df = df.sort_values("time_tag").reset_index(drop=True)
    df = df.fillna(method="ffill").fillna(method="bfill")

    df["speed"] = df["speed"].clip(200, 1000)
    df["density"] = df["density"].clip(0, 50)
    df["bz"] = df["bz"].clip(-20, 20)
    df["b"] = df["b"].clip(0, 30)
    
    return df


def scale_features(df: pd.DataFrame, feature_cols: list):
    """
    Apply MinMax scaling to selected features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_cols (list[str]): List of columns to scale.

    Returns:
        tuple: (scaled DataFrame, fitted MinMaxScaler object)
    """

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled, scaler


def make_supervised(df, feature_cols, target_col, window=24, horizon=3, indices=None):
    """
    Convert a time-series DataFrame into supervised learning windows for 
    sequence model (LSTM).

    Args:
        df (pd.DataFrame): Input data with features + target.
        feature_cols (list[str]): Feature column names.
        target_col (str): Target column name.
        window (int): Number of past timesteps for input.
        horizon (int): Number of future timesteps to predict.
        indices (np.array, optional): A pre-shuffled list of sample indices to generate. Defaults to sequential.

    Returns:
        tuple: (X, y) NumPy arrays ready for model training.
            - X shape: (num_samples, window, num_features)
            - y shape: (num_samples, horizon)
    """
    
    feature_values = df[feature_cols].values.astype(np.float32)
    target_values = df[target_col].values.astype(np.float32)
    
    num_samples = len(df) - window - horizon + 1
    if indices is None:
        indices = np.arange(num_samples)
    
    if len(indices) == 0:
        return np.array([]), np.array([])
        
    num_features = len(feature_cols)
    
    # Pre-allocate NumPy arrays to be memory efficient
    X = np.empty((len(indices), window, num_features), dtype=np.float32)
    y = np.empty((len(indices), horizon), dtype=np.float32)
    
    # Use array slicing to fill the pre-allocated arrays
    for i, sample_idx in enumerate(indices):
        X[i] = feature_values[sample_idx : sample_idx + window]
        y[i] = target_values[sample_idx + window : sample_idx + window + horizon]
        
    return X, y
