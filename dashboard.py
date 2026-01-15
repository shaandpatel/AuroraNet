import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import os
from datetime import datetime
import numpy as np

# --- CONFIGURATION ---
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# Page Config: "Wide" layout
st.set_page_config(page_title="Aurora Forecast", page_icon="🌌", layout="wide")

# --- CUSTOM CSS ---
# Tightening spacing and styling metrics as cards
st.markdown("""
    <style>
        /* Reduce top padding */
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        
        /* Style Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #1E1E1E;
            border: 1px solid #333;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.5);
        }
        
        /* Custom Header Font */
        h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 300; }
    </style>
""", unsafe_allow_html=True)

# --- DATA CONSTANTS ---
CITIES = {
    "Fairbanks, AK": {"coords": [64.84, -147.72], "mlat": 65.5}, 
    "Yellowknife, CA": {"coords": [62.45, -114.37], "mlat": 69.8},
    "Reykjavik, IS": {"coords": [64.14, -21.96],  "mlat": 64.9}, 
    "Tromsø, NO":    {"coords": [69.64, 18.95],   "mlat": 66.7},
    "Oslo, NO":      {"coords": [59.91, 10.75],   "mlat": 56.5}, 
    "Seattle, WA":   {"coords": [47.60, -122.33], "mlat": 53.8}, 
    "Toronto, CA":   {"coords": [43.65, -79.38],  "mlat": 54.0}, 
    "New York, NY":  {"coords": [40.71, -74.00],  "mlat": 50.5},
    "London, UK":    {"coords": [51.50, -0.12],   "mlat": 47.9}, 
}

# --- HELPER FUNCTIONS ---
def get_auroral_status(user_mlat, kp_value):
    boundary_mlat = 67 - (3 * kp_value)
    delta = user_mlat - boundary_mlat
    if delta >= 5: return "Very High", boundary_mlat, "#FF4B4B"  # Red
    elif 2 <= delta < 5: return "Good", boundary_mlat, "#FFA500" # Orange
    elif 0 <= delta < 2: return "Possible", boundary_mlat, "#FFFF00" # Yellow
    else: return "Low", boundary_mlat, "#A0A0A0" # Grey

def get_prediction():
    """Fetches data from backend with error handling"""
    try:
        response = requests.post(API_URL, json={}, timeout=8)
        if response.status_code == 200: return response.json()
    except Exception:
        return None
    return None

# --- SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    selected_city = st.selectbox("Observer Location", list(CITIES.keys()))
    user_data = CITIES[selected_city]
    
    st.divider()
    st.caption(f"Magnetic Lat: {user_data['mlat']}°")
    if st.button("Reload Data", use_container_width=True):
        st.cache_data.clear()

# --- MAIN APP LOGIC ---
if 'data' not in st.session_state or st.button("Refresh", key="hidden_refresh", help="Hidden trigger"):
    st.session_state['data'] = get_prediction()

data = st.session_state.get('data')

# Top Header Row
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("Geomagnetic Activity Monitor")
    st.caption("Real-time deep learning inference for auroral oval prediction.")

if data:
    timestamps = data['forecast_timestamps']
    kp_values = data['kp_predictions']
    
    # Process Timestamps
    ts_objects = pd.to_datetime(timestamps).to_pydatetime()
    formatted_times = [t.strftime('%H:%M UTC') for t in ts_objects]

    # --- 1. INTERACTIVE SLIDER (THE CONTROLLER) ---
    # We move this UP so it acts as the controller for everything below
    selected_time_str = st.select_slider(
        "Forecast Timeline", 
        options=formatted_times,
        value=formatted_times[0],
        label_visibility="collapsed" # Cleaner look
    )
    
    time_index = formatted_times.index(selected_time_str)
    selected_kp = kp_values[time_index]
    selected_ts = ts_objects[time_index]

    # --- 2. HEADS-UP DISPLAY (METRICS) ---
    status_text, boundary_mlat, status_color = get_auroral_status(user_data['mlat'], selected_kp)
    max_kp = max(kp_values)
    
    # Using 4 columns for a dashboard strip
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Time", selected_time_str)
    m2.metric("Predicted Kp", f"{selected_kp:.2f}")
    m3.metric("Auroral Boundary", f"{boundary_mlat:.1f}° MLAT")
    m4.markdown(f"""
        <div style="background-color: {status_color}; color: black; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">
            {status_text} Chance
        </div>
    """, unsafe_allow_html=True)

    # --- 3. MAIN VISUALIZATION ROW ---
    st.divider()
    col_map, col_chart = st.columns([3, 2])

    with col_map:
        st.subheader("Auroral Oval View")
        
        # Map Logic
        m = folium.Map(
            location=[50, -10], # Center on Atlantic for broader view
            zoom_start=2, 
            tiles="CartoDB dark_matter",
            control_scale=True,
            zoom_control=False # Cleaner
        )
        
        # User Location - Clean Circle instead of Icon
        folium.CircleMarker(
            location=user_data['coords'],
            radius=6,
            color="#FFFFFF",
            fill=True,
            fill_color="#3388ff",
            fill_opacity=1,
            tooltip=f"{selected_city} (MLAT: {user_data['mlat']})"
        ).add_to(m)

        # Auroral Boundary Line
        line_coords = []
        for lon in range(-180, 181, 5):
            if -130 <= lon <= -60: geo_lat = boundary_mlat - 9.5
            elif -10 <= lon <= 40: geo_lat = boundary_mlat + 3.5
            else: geo_lat = boundary_mlat - 2 
            line_coords.append([geo_lat, lon])

        # Smooth curve 
        folium.PolyLine(
            locations=line_coords,
            color="#00FFFF", # Cyan 
            weight=2,
            opacity=0.8,
            dash_array='5, 10', # Dashed line implies "Forecast/boundary"
            tooltip=f"Visibility Boundary (Kp {selected_kp:.1f})"
        ).add_to(m)

        st_folium(m, height=450, width="100%")

    with col_chart:
        st.subheader("Activity Trend (1 Hour)")
        
        fig = go.Figure()
        
        # Area chart 
        fig.add_trace(go.Scatter(
            x=ts_objects, y=kp_values,
            fill='tozeroy',
            mode='lines',
            line=dict(color='#00FFFF', width=2),
            fillcolor='rgba(0, 255, 255, 0.1)', # Subtle glow
            name='Kp Index'
        ))

        # Current Time Marker
        fig.add_vline(x=selected_ts, line_width=1, line_dash="solid", line_color="white")
        
        # Minimalist Layout
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(showgrid=True, gridcolor='#333', range=[0, 9]),
            xaxis=dict(showgrid=False),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 4. FOOTER INFO ---
    if data.get("source"):
        st.caption(f"Data Source: {data['source']} | Last Updated: {data.get('cached_at', 'N/A')}")
        if data.get("warning"):
            st.warning(data['warning'])

else:
    # Empty State (Loading)
    st.info("Initializing model connection...")