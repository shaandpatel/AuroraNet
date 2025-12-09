"""
Fetch real-time solar wind data from NOAA SWPC (DSCOVR).

Endpoints:
    speed.json
    density.json
    mag-field-bz.json
    mag-field-total.json
"""

import requests
import pandas as pd
from functools import reduce

REALTIME_SW_URL = "https://services.swpc.noaa.gov/products/solar-wind/"


def fetch_realtime_solarwind() -> pd.DataFrame:
    """
    Fetch real-time solar-wind measurements from NOAA’s DSCOVR feeds.

    Combines speed, density, and magnetic field components into a single
    time-aligned DataFrame ready for inference.

    Returns:
        pd.DataFrame: ['time_tag', 'speed', 'density', 'b', 'bz', 'temp'].
    """

    endpoints = {
        "plasma": ["density", "speed", "temperature"],
        "mag": ["bt", "bz_gsm"]
    }

    dfs = []
    for product, columns in endpoints.items():
        r = requests.get(f"{REALTIME_SW_URL}{product}-3-day.json", timeout=10)
        r.raise_for_status()
        raw = r.json()

        df = pd.DataFrame(raw[1:], columns=raw[0])
        df["time_tag"] = pd.to_datetime(df["time_tag"])
        
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        dfs.append(df[["time_tag"] + columns])

    # Merge using nearest timestamps
    merged = reduce(lambda left, right: pd.merge_asof(left, right, on="time_tag"), dfs)
    
    # Rename to match historical data
    merged.rename(columns={
        "bt": "b",
        "bz_gsm": "bz",
        "temperature": "temp"
    }, inplace=True)

    # Select final columns
    final_cols = ["time_tag", "speed", "density", "b", "bz", "temp"]
    return merged[final_cols].dropna().reset_index(drop=True)
