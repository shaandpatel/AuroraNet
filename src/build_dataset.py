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
import numpy as np
import pandas as pd

from src.data import (
    fetch_omni_data,
    fetch_kp_range,
    clean_solarwind,
    add_time_features,
    add_moving_averages,
    scale_features,
    make_supervised
)

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main(start_year, end_year, output_dir, test_split_ratio, window, horizon, resolution):
    """Main function to run the data preparation pipeline."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------
    # 1. Fetch Historical Data
    # ---------------------------
    logging.info(f"Fetching historical data from {start_year} to {end_year}...")
    sw_df = fetch_omni_data(start_year=start_year, end_year=end_year, resolution=resolution)
    kp_df = fetch_kp_range(start_year=start_year, end_year=end_year)

    # --- Add validation check ---
    if sw_df.empty or kp_df.empty:
        logging.error("One or both initial dataframes are empty. Aborting. Check data fetching for the specified year range.")
        return

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
    kp_df = kp_df.set_index('time_tag').resample(f'{resolution}min').ffill()
    
    # Merge the two dataframes on the time_tag index
    df_merged = pd.merge(sw_df, kp_df, on='time_tag', how='inner')
    logging.info(f"Merged dataframe shape: {df_merged.shape}")

    # --- Add validation check ---
    if df_merged.empty:
        logging.error("Merged dataframe is empty. There are no overlapping timestamps between solar wind and Kp data. Aborting.")
        return

    # ---------------------------
    # 3. Clean and Feature Engineer
    # ---------------------------
    logging.info("Cleaning data and engineering features...")
    df_clean = clean_solarwind(df_merged)
    df_feat = add_time_features(df_clean)
    
    # --- Create Lagged Kp Feature ---
    # Shift Kp by 1 to ensure we are using strictly past data for the current timestep's feature.
    # This prevents the raw target 'kp_index' from leaking into the input features.
    df_feat['kp_prev'] = df_feat['kp_index'].shift(1)
    df_feat = df_feat.dropna()  # Drop the first row which is now NaN

    # Use 'kp_prev' as a feature, but EXCLUDE the target 'kp_index'
    feature_cols = [col for col in df_feat.columns if col not in ['time_tag', 'kp_index']]
    target_col = 'kp_index'
    logging.info(f"Using features: {feature_cols}")

    # ---------------------------
    # 4. Split Data into Train and Test sets (Time-based)
    # ---------------------------
    split_index = int(len(df_feat) * (1 - test_split_ratio))
    df_train = df_feat.iloc[:split_index]
    df_test = df_feat.iloc[split_index:]
    logging.info(f"Splitting data: {len(df_train)} training samples, {len(df_test)} testing samples.")

    # ---------------------------
    # 4. Scale Features
    # ---------------------------
    logging.info("Scaling features (fitting on training data only)...")
    # Fit the scaler ONLY on the training data to prevent data leakage
    df_train_scaled, scaler = scale_features(df_train, feature_cols)
    
    # Transform the test data using the same scaler
    df_test_scaled = df_test.copy()
    df_test_scaled[feature_cols] = scaler.transform(df_test[feature_cols])
    
    # ---------------------------
    # 5. Add Moving Averages (on scaled data)
    # ---------------------------
    ma_cols = ['speed', 'density', 'bz', 'b']
    logging.info(f"Adding moving averages for columns: {ma_cols}")
    df_train_final = add_moving_averages(df_train_scaled, columns=ma_cols, window=10)
    df_train_final = add_moving_averages(df_train_final, columns=ma_cols, window=50)
    df_test_final = add_moving_averages(df_test_scaled, columns=ma_cols, window=10)
    df_test_final = add_moving_averages(df_test_final, columns=ma_cols, window=50)

    # Redefine feature columns to include the new moving average features
    final_feature_cols = [col for col in df_train_final.columns if col not in ['kp_index', 'time_tag']]

    # Save the scaler for use during inference
    scaler_path = Path("models") / "scaler.pkl"
    scaler_path.parent.mkdir(exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")

    # ---------------------------
    # 6. Create Supervised Sequences
    # ---------------------------
    logging.info(f"Creating supervised sequences with window={window} and horizon={horizon}...")
    X_train, y_train = make_supervised(df_train_final, final_feature_cols, target_col, window=window, horizon=horizon)
    X_test, y_test = make_supervised(df_test_final, final_feature_cols, target_col, window=window, horizon=horizon)
    logging.info(f"Training set shapes: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Test set shapes: X={X_test.shape}, y={y_test.shape}")

    # --- Add final validation check ---
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        logging.error("Created datasets are empty. This can happen if the time range is too small or data is missing.")
        logging.error("Aborting to prevent errors in the training pipeline.")
        return

    # ---------------------------
    # 7. Save Final Datasets
    # ---------------------------
    train_path = output_path / "training_data.npz"
    test_path = output_path / "test_data.npz"

    logging.info(f"Saving training data to {train_path} (uncompressed for speed)...")
    np.savez(train_path, X=X_train, y=y_train)

    logging.info(f"Saving test data to {test_path} (uncompressed for speed)...")
    np.savez(test_path, X=X_test, y=y_test)

    logging.info("Data preparation pipeline complete!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full data preparation pipeline for the Aurora project.")
    parser.add_argument("--start", type=int, default=2010, help="Start year for historical data.")
    parser.add_argument("--end", type=int, default=2020, help="End year for historical data.")
    parser.add_argument("--output", type=str, default="datafiles/processed", help="Directory to save processed data and artifacts.")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction of data to use for the test set.")
    parser.add_argument("--window", type=int, default=72, help="Number of past time steps to use as input features.")
    parser.add_argument("--horizon", type=int, default=6, help="Number of future time steps to predict.")
    parser.add_argument("--resolution", type=int, default=24, help="Data resolution in minutes.")
    args = parser.parse_args()

    main(args.start, args.end, args.output, args.test_split, args.window, args.horizon, args.resolution)