import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path to allow importing app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, data_cache

# --- FIXTURES ---

@pytest.fixture
def mock_solar_data():
    """Generates dummy solar wind dataframe."""
    return pd.DataFrame({
        'time_tag': pd.date_range(start=datetime.now(), periods=10, freq='min'),
        'speed': [400.0] * 10,
        'density': [5.0] * 10,
        'bz': [-2.0] * 10
    })

@pytest.fixture
def mock_kp_data():
    """Generates dummy Kp index dataframe."""
    return pd.DataFrame({
        'time_tag': pd.date_range(start=datetime.now(), periods=5, freq='3H'),
        'kp_index': [2.0] * 5
    })

@pytest.fixture
def mock_dependencies(mock_solar_data, mock_kp_data):
    """
    Patches all external dependencies (Model, NOAA API).
    Returns the mock objects so we can assert they were called.
    """
    # Reset global cache to ensure test isolation
    data_cache["data"] = None
    data_cache["last_updated"] = None
    data_cache["source"] = None

    with patch("app.load_model_from_artifact") as mock_load, \
         patch("app.fetch_realtime_solarwind") as mock_sw, \
         patch("app.fetch_recent_kp") as mock_kp, \
         patch("app.predict_kp") as mock_pred:
        
        # Setup default successful returns
        mock_load.return_value = (MagicMock(), MagicMock(), {"seq_length": 60})
        mock_sw.return_value = mock_solar_data
        mock_kp.return_value = mock_kp_data
        mock_pred.return_value = np.array([3.0, 3.5, 4.0])
        
        yield {
            "load": mock_load,
            "sw": mock_sw,
            "kp": mock_kp,
            "pred": mock_pred
        }

@pytest.fixture
def client(mock_dependencies):
    """Returns a TestClient with mocked dependencies active."""
    # TestClient triggers the lifespan (startup) event where model loads
    with TestClient(app) as c:
        yield c

# --- TESTS ---

def test_health_check(client):
    """Ensure the health endpoint returns 200 and status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "active"

def test_predict_endpoint_success(client, mock_dependencies):
    """Test the full prediction flow with mocked data."""
    response = client.post("/predict")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "forecast_timestamps" in data
    assert "kp_predictions" in data
    assert len(data["kp_predictions"]) == 3
    assert data["source"] == "NOAA Real-time"
    
    # Verify our mocks were actually called
    assert mock_dependencies["sw"].called
    assert mock_dependencies["pred"].called

def test_caching_behavior(client, mock_dependencies):
    """Test that subsequent requests use the cache instead of fetching new data."""
    # First call: Should fetch data
    client.post("/predict")
    assert mock_dependencies["sw"].call_count == 1
    
    # Second call: Should use cache (fetch count should NOT increase)
    client.post("/predict")
    assert mock_dependencies["sw"].call_count == 1

def test_model_loading_failure():
    """Test that the API handles model loading failures gracefully."""
    with patch("app.load_model_from_artifact") as mock_load:
        mock_load.return_value = (None, None, None) # Simulate failure
        
        with TestClient(app) as c:
            response = c.post("/predict")
            assert response.status_code == 503
            assert "Model is not loaded" in response.json()["detail"]