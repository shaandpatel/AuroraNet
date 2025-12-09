import os
import logging
import pandas as pd
import pyomnidata

# ---------------------------
# Logging setup
# ---------------------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def fetch_omni_data(start_year: int, end_year: int, resolution: int = 1) -> pd.DataFrame:
    """
    Download historical OMNI solar-wind data for model training.

    Provides key solar-wind features: speed, density, Bz, and Bt.

    Args:
        year_start (int): Starting year.
        year_end (int): Ending year (inclusive).
        resolution (int, optional): Time resolution in minutes. Default is 60 (hourly).

    Returns:
        pd.DataFrame: ['time_tag', 'speed', 'density', 'b', 'bz', 'temp'].
    """
    logger.info(f"Fetching OMNI {resolution}-min data {start_year}-{end_year}...")

    # Actual fetch via pyomnidata
    raw = pyomnidata.GetOMNI([start_year, end_year], Res=resolution)
    df = pd.DataFrame.from_records(raw)

    # Create a proper datetime index from Date and UT
    df['time_tag'] = pd.to_datetime(df['Date'], format='%Y%m%d') + pd.to_timedelta(df['ut'], unit='D')
    df['time_tag'] = df['time_tag'].dt.round('S')  # Round to nearest second

    df = df.rename(columns={
        "FlowSpeed": "speed",
        "ProtonDensity": "density",
        "B": "b",
        "BzGSM": "bz",
        "Temp": "temp",
    })
    
    # Set time_tag as index and resample to ensure consistent 1-minute frequency
    df = df.set_index('time_tag')
    df = df[["speed", "density", "b", "bz", "temp"]]
    df = df.resample(f'{resolution}min').mean()  # Resample to the desired frequency
    df = df.interpolate(method='linear').bfill()  # Interpolate and then back-fill any remaining NaNs
    
    logger.info(f"Fetched {len(df)} rows of OMNI data")
    return df.reset_index()
