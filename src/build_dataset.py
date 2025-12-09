"""
Master script for building the training dataset.

This script orchestrates the entire data pipeline:
1. Fetches historical solar wind (OMNI) and Kp index data.
2. Merges the two datasets, handling different time resolutions.
3. Cleans the data (fills missing values, clips outliers).
4. Engineers features (cyclical time, moving averages).
5. Scales the features and saves the scaler object.
6. Creates supervised learning windows (sequences).
7. Saves the final X and y arrays for model training.
"""

import logging
import argparse
from pathlib import Path
import joblib
import pandas as pd

from src.data import (
    fetch_omni_data,
    fetch_kp_range,
    clean_solarwind,
    add_time_features,
    add_moving_averages,
    scale_features,
    make_supervised,
)

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main(start_year, end_year, output_dir):
    """Main function to run the data preparation pipeline."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------
    # 1. Fetch Historical Data
    # ---------------------------
    logging.info(f"Fetching historical data from {start_year} to {end_year}...")
    sw_df = fetch_omni_data(start_year=start_year, end_year=end_year, resolution=1)
    kp_df = fetch_kp_range(start_year=start_year, end_year=end_year)

    # --- Pre-emptive fix for duplicate labels ---
    # Group by time_tag and average to remove any duplicate timestamps before merging.
    logging.info("Ensuring timestamp uniqueness before merging...")
    sw_df = sw_df.groupby('time_tag').mean().reset_index()
    kp_df = kp_df.groupby('time_tag').mean().reset_index()
    
    # ---------------------------
    # 2. Merge Data
    # ---------------------------
    logging.info("Merging solar wind and Kp index data...")
    # Kp is 3-hourly, so we need to resample it to match the 1-minute solar wind data.
    # We forward-fill because the Kp index is constant over its 3-hour window.
    kp_df = kp_df.set_index('time_tag').resample('1min').ffill()
    
    # Merge the two dataframes on the time_tag index
    df_merged = pd.merge(sw_df, kp_df, on='time_tag', how='inner')
    logging.info(f"Merged dataframe shape: {df_merged.shape}")

    # ---------------------------
    # 3. Clean and Feature Engineer
    # ---------------------------
    logging.info("Cleaning data and engineering features...")
    df_clean = clean_solarwind(df_merged)
    df_feat = add_time_features(df_clean)
    
    # Define feature columns (everything except the target and original timestamp)
    feature_cols = [col for col in df_feat.columns if col not in ['kp_index', 'time_tag']]
    target_col = 'kp_index'
    logging.info(f"Using features: {feature_cols}")

    # ---------------------------
    # 4. Scale Features
    # ---------------------------
    logging.info("Scaling features...")
    df_scaled, scaler = scale_features(df_feat, feature_cols)
    
    # Save the scaler for use during inference
    scaler_path = output_path / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")

    # ---------------------------
    # 5. Create Supervised Sequences
    # ---------------------------
    logging.info("Creating supervised learning sequences...")
    X, y = make_supervised(df_scaled, feature_cols, target_col, window=72, horizon=6)
    logging.info(f"Created sequences with X shape: {X.shape} and y shape: {y.shape}")

    # ---------------------------
    # 6. Save Final Datasets
    # ---------------------------
    X_path = output_path / "X_train.npy"
    y_path = output_path / "y_train.npy"
    pd.to_pickle(X, X_path)
    pd.to_pickle(y, y_path)
    logging.info(f"Training data saved to {X_path} and {y_path}")
    logging.info("Data preparation pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full data preparation pipeline for the Aurora project.")
    parser.add_argument("--start", type=int, default=2010, help="Start year for historical data.")
    parser.add_argument("--end", type=int, default=2020, help="End year for historical data.")
    parser.add_argument("--output", type=str, default="data/processed", help="Directory to save processed data and artifacts.")
    args = parser.parse_args()

    main(args.start, args.end, args.output)