"""
Fetching Kp index data from NOAA SWPC.
These functions are used for inference (real time data).
"""

import requests
import pandas as pd

REALTIME_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"


def fetch_recent_kp(hours: int = 72) -> pd.DataFrame:
    """
    Retrieve recent real-time Kp index from NOAA’s live JSON feed.

    Can fetch the last `hours` of data for immediate inference or visualization.

    Args:
        hours (int, optional): Number of past hours to retrieve. Default is 72.

    Returns:
        pd.DataFrame: ['time_tag', 'kp_index'] with recent Kp values.
    """

    r = requests.get(REALTIME_KP_URL, timeout=10)
    r.raise_for_status()

    raw = r.json()
    df = pd.DataFrame(raw[1:], columns=raw[0])
    df.rename(columns={'Kp': 'kp_index'}, inplace=True)

    df['time_tag'] = pd.to_datetime(df['time_tag'])
    df['kp_index'] = pd.to_numeric(df['kp_index'], errors='coerce')

    recent = df[df['time_tag'] >= df['time_tag'].max() - pd.Timedelta(hours=hours)]
    return recent[['time_tag', 'kp_index']].reset_index(drop=True)