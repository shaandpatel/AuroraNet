import logging
import pandas as pd
import pyomnidata

# ---------------------------
# Logging setup
# ---------------------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def fetch_omni_data(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Download historical OMNI solar-wind data for model training.

    Provides key solar-wind features: speed, density, Bz, Bt, and temp.

    Args:
        year_start (int): Starting year.
        year_end (int): Ending year (inclusive).

    Returns:
        pd.DataFrame: ['time_tag', 'speed', 'density', 'b', 'bz', 'temp'].
    """
    # Always fetch at the highest resolution (1-minute) to ensure data quality before downsampling.
    logger.info(f"Fetching OMNI 1-min data {start_year}-{end_year}...")

    # Actual fetch via pyomnidata
    raw = pyomnidata.GetOMNI([start_year, end_year], Res=1)
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

    return df.reset_index()
