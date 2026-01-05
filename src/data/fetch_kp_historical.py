"""
Fetching Kp index data from NOAA SWPC.
These functions are used for training (historical data).

Each year is available at:
    https://services.swpc.noaa.gov/json/planetary_k_index_YYYY.json
"""


import re
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def fetch_kp_year(file_year: int) -> pd.DataFrame:
    """
    Read a Kp TXT file (e.g. '2010_DGD.txt')
    and return NOAA-style Kp time series.
    """
    
    file_path = Path(f"./datafiles/kpdata/{file_year}_DGD.txt")
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    rows = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # Ensure the line starts with YYYY MM DD
            if not re.match(r"^\d{4}\s+\d{2}\s+\d{2}", line):
                continue

            # Extract all numbers, even glued: [-1-1] → [-1, -1]
            nums = list(map(int, re.findall(r"-?\d+", line)))

            # nums = [YYYY, MM, DD, A_index, K1, K2, ..., K8]
            yyyy, mm, dd = nums[:3]
            base_date = datetime(yyyy, mm, dd)

            # Last 8 numbers = planetary Kp values
            kp_values = nums[-8:]

            # Expand each Kp value into 3-hour timestamps
            for i, kp in enumerate(kp_values):
                timestamp = base_date + timedelta(hours=3 * i)
                rows.append([timestamp, float(kp)])
    
    df = pd.DataFrame(rows, columns=["time_tag", "kp_index"])
    return df





def fetch_kp_range(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Combine multiple TXT Kp files into one continuous time series.
    """

    frames = []
    for y in range(start_year, end_year + 1):
        print(f"Loading Kp for {y} ...")
        frames.append(fetch_kp_year(y))

    return pd.concat(frames).sort_values("time_tag").reset_index(drop=True)
