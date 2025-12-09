import pandas as pd
import numpy as np

def add_moving_averages(df, columns, window=3):
    """
    Add moving average columns to smooth short-term variations.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list[str]): Columns to smooth
        window (int): Window size in timesteps
    
    Returns:
        pd.DataFrame: DataFrame with new '_ma' columns
    """
    df_copy = df.copy()
    for col in columns:
        # Calculate the moving average. This will create NaNs at the beginning.
        ma_col = f"{col}_ma{window}"
        df_copy[ma_col] = df_copy[col].rolling(window, min_periods=1).mean()
    return df_copy


def add_time_features(df, time_col="time_tag"):
    """
    Add cyclical time features (hour of day, day of year) for sequence models.
    
    Args:
        df (pd.DataFrame): Input DataFrame with datetime column
        time_col (str): Name of the datetime column
    
    Returns:
        pd.DataFrame: DataFrame with new 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    """
    df_copy = df.copy()
    dt = pd.to_datetime(df_copy[time_col])
    df_copy["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df_copy["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df_copy["day_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365)
    df_copy["day_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365)
    df_copy['month_sin'] = np.sin(2 * np.pi * dt.dt.month/12)
    df_copy['month_cos'] = np.cos(2 * np.pi * dt.dt.month/12)
    df_copy['year'] = dt.dt.year - 2000
    
    return df_copy
