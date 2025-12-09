"""
Test the full data → model pipeline locally
"""

import pandas as pd
from src.data import (
    fetch_omni_data,
    fetch_kp_range,
    fetch_recent_kp,
    fetch_realtime_solarwind,
    clean_solarwind,
    scale_features,
    make_supervised,
    add_moving_averages,
    add_time_features,
)
from src.models.lstm_model import KpLSTM
import torch
import joblib

# ---------------------------
# Step 1: Fetch data
# ---------------------------
print("Fetching example solar wind data...")
sw_df = fetch_omni_data(start_year=2010, end_year=2011, resolution=1)
# kp_df = fetch_kp_range(start_year=2010, end_year=2011)

# kp_rt = fetch_recent_kp()
# sw_rt = fetch_realtime_solarwind()

# # ---------------------------
# # Step 2: Preprocess
# # ---------------------------
print("Cleaning solar wind data...")
sw_clean = clean_solarwind(sw_df)

print("Adding features...")
sw_feat = add_time_features(sw_clean)

# print("Scaling features...")
# scaler = joblib.load("models/scaler.pkl")  # must match training scaler
# sw_scaled = scaler.transform(sw_feat)

# # ---------------------------
# # Step 3: Make supervised sequence
# # ---------------------------
# seq_length = 24
# X_seq = make_supervised(sw_scaled, seq_length=seq_length)

# # ---------------------------
# # Step 4: Load model and predict
# # ---------------------------
# model = KpLSTM(input_size=X_seq.shape[2], hidden_size=64, num_layers=2)
# model.load_state_dict(torch.load("models/saved_model.pth", map_location="cpu"))
# model.eval()

# X_tensor = torch.tensor(X_seq[-1:], dtype=torch.float32)  # last sequence
# with torch.no_grad():
#     kp_pred = model(X_tensor)

# print("Predicted Kp:", kp_pred.numpy().flatten())
