import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    clean_solarwind,
    add_time_features,
    add_moving_averages,
    make_supervised,
    scale_features
)

# --- FIXTURES ---

@pytest.fixture
def sample_raw_df():
    """Creates a raw dataframe with some issues (NaNs, outliers) to test cleaning."""
    return pd.DataFrame({
        'time_tag': pd.date_range(start='2023-01-01', periods=5, freq='H'),
        'speed': [300.0, 400.0, 1200.0, np.nan, 400.0],  # 1200 is outlier, NaN needs filling
        'density': [5.0, 5.0, 5.0, 5.0, 5.0],
        'bz': [0.0, 0.0, 0.0, 0.0, 0.0],
        'b': [5.0, 5.0, 5.0, 5.0, 5.0],
        'temp': [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    })

@pytest.fixture
def sample_clean_df():
    """Creates a clean dataframe for feature engineering tests."""
    return pd.DataFrame({
        'time_tag': pd.date_range(start='2023-01-01', periods=10, freq='H'),
        'val': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'target': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    })

# --- UNIT TESTS ---

def test_clean_solarwind_logic(sample_raw_df):
    """Test that outliers are clipped and NaNs are filled."""
    cleaned = clean_solarwind(sample_raw_df)
    
    # Check outlier clipping (speed > 1000 should be clipped to 1000)
    assert cleaned['speed'].max() <= 1000.0
    
    # Check NaN filling
    assert not cleaned.isnull().values.any()
    
    # Check sorting (though input was sorted, function ensures it)
    assert pd.api.types.is_datetime64_any_dtype(cleaned['time_tag'])

def test_add_time_features(sample_clean_df):
    """Test that cyclical time features are added correctly."""
    df = add_time_features(sample_clean_df)
    
    expected_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    for col in expected_cols:
        assert col in df.columns
        # Sin/Cos values must be strictly between -1 and 1
        assert df[col].min() >= -1.0
        assert df[col].max() <= 1.0

def test_add_moving_averages(sample_clean_df):
    """Test moving average calculation."""
    window = 3
    df = add_moving_averages(sample_clean_df, columns=['val'], window=window)
    
    col_name = f"val_ma{window}"
    assert col_name in df.columns
    
    # Check calculation at index 2: (1+2+3)/3 = 2.0
    assert df[col_name].iloc[2] == 2.0
    # Check calculation at index 3: (2+3+4)/3 = 3.0
    assert df[col_name].iloc[3] == 3.0

def test_scale_features(sample_clean_df):
    """Test MinMax scaling."""
    cols = ['val']
    scaled_df, scaler = scale_features(sample_clean_df, cols)
    
    # Check if scaler was fitted
    assert hasattr(scaler, 'scale_')
    
    # Check range (MinMax defaults to 0-1)
    assert scaled_df['val'].min() == 0.0
    assert scaled_df['val'].max() == 1.0

def test_make_supervised():
    """Test sequence generation for LSTM."""
    # Create simple sequential data
    data = np.array([
        [10, 1],
        [20, 2],
        [30, 3],
        [40, 4],
        [50, 5],
        [60, 6]
    ])
    df = pd.DataFrame(data, columns=['feature', 'target'])
    
    window = 2
    horizon = 1
    
    X, y = make_supervised(df, ['feature'], 'target', window=window, horizon=horizon)
    
    # Total samples = 6. Window=2, Horizon=1.
    # Valid samples = 6 - 2 - 1 + 1 = 4 samples.
    assert len(X) == 4
    assert len(y) == 4
    
    # Check shapes
    assert X.shape == (4, 2, 1) # (samples, window, features)
    assert y.shape == (4, 1)    # (samples, horizon)
    
    # Check content of first sample
    # X[0] should be input [10, 20]
    assert X[0, 0, 0] == 10
    assert X[0, 1, 0] == 20
    # y[0] should be target [3] (the value immediately following the window)
    assert y[0, 0] == 3.0
