import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- MOCK STREAMLIT ---
# Mock streamlit modules before importing dashboard.py
# because dashboard.py runs top-level code (st.set_page_config, etc.) on import.
mock_st = MagicMock()
# Fix for KeyError: st.selectbox returns a Mock by default, which isn't in the CITIES dict.
mock_st.selectbox.return_value = "Fairbanks, AK"
# Fix for TypeError: st.session_state.get returns a Mock, causing logic to run with Mocks.
# We use a real dict so .get() returns None (or what we put in), skipping the UI logic.
mock_st.session_state = {}
sys.modules["streamlit"] = mock_st

# Mock requests to prevent network calls and ensure get_prediction returns None
mock_requests = MagicMock()
mock_requests.exceptions.ConnectionError = ConnectionError
mock_requests.exceptions.Timeout = TimeoutError
sys.modules["requests"] = mock_requests
sys.modules["streamlit_folium"] = MagicMock()
sys.modules["folium"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()

# Now it is safe to import the function we want to test
from dashboard import get_auroral_status

def test_auroral_status_logic():
    """
    Test the physics-based visibility logic.
    Formula: Boundary_MLAT = 67 - 3 * Kp
    """
    
    # Case 1: High Kp (Storm), User at high latitude
    # MLAT 65, Kp 8 -> Boundary = 67 - 24 = 43. Delta = 65 - 43 = 22.
    # Delta >= 5 -> "Very High Chance"
    status, msg, boundary = get_auroral_status(user_mlat=65, kp_value=8)
    assert status == "Very High Chance"
    assert boundary == 43
    
    # Case 2: Low Kp (Quiet), User at mid latitude (e.g., NYC)
    # MLAT 50, Kp 2 -> Boundary = 67 - 6 = 61. Delta = 50 - 61 = -11.
    # Delta < -2 -> "Not Expected"
    status, msg, boundary = get_auroral_status(user_mlat=50, kp_value=2)
    assert status == "Not Expected"
    
    # Case 3: Moderate Kp, User near boundary
    # MLAT 60, Kp 3 -> Boundary = 67 - 9 = 58. Delta = 60 - 58 = 2.
    # 2 <= Delta < 5 -> "Good Chance"
    status, msg, boundary = get_auroral_status(user_mlat=60, kp_value=3)
    assert status == "Good Chance"