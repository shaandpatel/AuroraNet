import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import os
from datetime import datetime

# --- CONFIGURATION ---
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Aurora Forecast", page_icon="🌌", layout="wide")

# Database of Cities (Approx Magnetic Latitudes)
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

def get_auroral_status(user_mlat, kp_value):
    """
    Calculates visibility based on User MLAT and a specific Kp value.
    Boundary Formula: MLAT ≈ 67 - 3 * Kp
    """
    boundary_mlat = 67 - (3 * kp_value)
    delta = user_mlat - boundary_mlat
    
    if delta >= 5:
        return "Very High Chance", "Aurora likely overhead.", boundary_mlat
    elif 2 <= delta < 5:
        return "Good Chance", "Aurora likely visible.", boundary_mlat
    elif 0 <= delta < 2:
        return "Possible", "Look towards the northern horizon.", boundary_mlat
    elif -2 <= delta < 0:
        return "Unlikely", "Only possible during extreme events.", boundary_mlat
    else:
        return "Not Expected", "The auroral oval is too far north.", boundary_mlat

# --- UI LAYOUT ---
st.title("🌌 Real-Time Aurora Forecast")
st.markdown("Physics-based deep learning model predicting geomagnetic activity (Kp Index).")

# Sidebar
with st.sidebar:
    st.header("Controls")
    if st.button("Refresh Forecast", type="primary"):
        st.session_state['refresh'] = True
    
    st.divider()
    st.markdown("### Location Settings")
    selected_city = st.selectbox("Select City:", list(CITIES.keys()))
    user_data = CITIES[selected_city]

# Fetch Logic
def get_prediction():
    try:
        with st.spinner("Fetching NOAA data & running inference..."):
            response = requests.post(API_URL, json={})
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.text}")
                return None
    except Exception:
        st.error("Could not connect to API. Is 'uvicorn app:app' running?")
        return None

if 'data' not in st.session_state or st.session_state.get('refresh', False):
    st.session_state['data'] = get_prediction()
    st.session_state['refresh'] = False

data = st.session_state.get('data')

if data:
    # Prepare Data
    timestamps = data['forecast_timestamps']
    kp_values = data['kp_predictions']
    
    # 1. Convert to native Python datetime objects
    ts_objects = pd.to_datetime(timestamps).to_pydatetime()
    
    formatted_times = [t.strftime('%H:%M %p (UTC)') for t in ts_objects]
    
    # --- FORECAST SLIDER ---
    st.markdown("### Forecast Timeline")
    st.info("Drag the slider to see the Aurora probability for different times in the future.")
    
    selected_time_str = st.select_slider(
        "Select Forecast Time:", 
        options=formatted_times,
        value=formatted_times[0]
    )
    
    # Get index and specific datetime object
    time_index = formatted_times.index(selected_time_str)
    selected_kp = kp_values[time_index]
    selected_ts = ts_objects[time_index]

    # --- METRICS ---
    max_kp = max(kp_values)
    max_kp_idx = kp_values.index(max_kp)
    peak_time_str = formatted_times[max_kp_idx]

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Kp at {selected_time_str}", f"{selected_kp:.2f}")
    col2.metric("Peak Activity Window", f"{max_kp:.2f}", help=f"Best chance: {peak_time_str}")
    
    status_header, status_text, boundary_mlat = get_auroral_status(user_data['mlat'], selected_kp)
    col3.metric("Visibility Status", status_header.split(" ")[0]) 

    # --- MAP ---
    st.divider()
    col_map, col_info = st.columns([2, 1])
    
    with col_info:
        st.subheader(f"📍 {selected_city}")
        st.markdown(f"**Forecast for:** `{selected_time_str}`")
        st.info(f"**{status_header}**\n\n{status_text}")
        
        if selected_kp < max_kp:
            st.warning(f"💡 Better chance at **{peak_time_str}** (Kp {max_kp:.2f})")
        
        with st.expander("Physics Calculation"):
            st.write(f"**User MLAT:** {user_data['mlat']}°")
            st.write(f"**Auroral Boundary:** {boundary_mlat:.1f}° MLAT")
            st.latex(r"\text{Boundary} \approx 67 - 3 \cdot Kp")
            st.write(f"**Delta:** {user_data['mlat'] - boundary_mlat:.2f}°")

    with col_map:
        m = folium.Map(location=user_data['coords'], zoom_start=3, tiles="CartoDB dark_matter")
        
        folium.Marker(
            location=user_data['coords'],
            tooltip=selected_city,
            icon=folium.Icon(color="blue", icon="user")
        ).add_to(m)
        
        # --- VISUALIZATION LOGIC ---
        # The relationship between Mag Lat and Geo Lat changes by Longitude.
        # We calculate the "Green Line" points dynamically to show the tilt.
        
        line_coords = []
        for lon in range(-180, 181, 5):
            # Approximate shift based on longitude
            # N. America (Lon -120 to -60): Mag is ~10 deg higher -> Line shifts South (-10)
            # Europe (Lon 0 to 30): Mag is ~3-5 deg lower -> Line shifts North (+4)
            
            if -130 <= lon <= -60: # Americas
                geo_lat = boundary_mlat - 9.5
            elif -10 <= lon <= 40: # Europe
                geo_lat = boundary_mlat + 3.5
            else: # Transition zones (Atlantic/Pacific)
                geo_lat = boundary_mlat - 2 # Average fallback
            
            line_coords.append([geo_lat, lon])

        folium.PolyLine(
            locations=line_coords,
            color="#39FF14", 
            weight=3,
            tooltip=f"Auroral Boundary (Kp {selected_kp:.1f})"
        ).add_to(m)
        
        st_folium(m, height=400, use_container_width=True)

    # --- 4. TIME SERIES CHART ---
    st.subheader("Kp Forecast Trend")
    fig = go.Figure()
    
    # Main Line
    fig.add_trace(go.Scatter(
        x=ts_objects, 
        y=kp_values, 
        mode='lines+markers', 
        name='Forecast', 
        line=dict(color='#00CC96')
    ))
    
    # Storm Threshold
    fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="G1 Storm")
    
    fig.add_shape(
        type="line",
        x0=selected_ts, y0=0,
        x1=selected_ts, y1=1,
        xref="x", yref="paper", # "paper" means y1=1 is the top of the chart
        line=dict(color="white", width=2, dash="dot")
    )

    # Add annotation manually since we aren't using add_vline
    fig.add_annotation(
        x=selected_ts, y=8.5,
        text="Selected",
        showarrow=False,
        font=dict(color="white")
    )
    
    fig.update_layout(template="plotly_dark", yaxis=dict(range=[0, 9]), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Waiting for data... Ensure backend is running.")