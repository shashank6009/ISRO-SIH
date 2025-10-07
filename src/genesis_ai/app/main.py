import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import time

from genesis_ai.inference.predictor_client import GenesisClient
from genesis_ai.integration.env_feed import fetch_space_weather, fetch_ionosphere_data, get_ground_stations, fetch_solar_activity, EnvironmentalMonitor

# --- UI Theme ---
st.set_page_config(
    page_title="GENESIS-AI | ISRO Mission Control",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced ISRO Mission Control Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global Overrides */
.stApp {
    background: linear-gradient(135deg, #0b0f23 0%, #1a1f3a 100%);
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}

/* Top Navigation Bar */
.genesis-topbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: linear-gradient(90deg, #0e1330 0%, #1a2040 100%);
    border-bottom: 2px solid rgba(0, 201, 255, 0.3);
    padding: 0.8rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    z-index: 1000;
    backdrop-filter: blur(10px);
}

.genesis-logo {
    display: flex;
    align-items: center;
    font-family: 'Roboto Mono', monospace;
    font-weight: 700;
    font-size: 1.4rem;
    color: #00c9ff;
    text-shadow: 0 0 10px rgba(0, 201, 255, 0.5);
}

.genesis-logo::before {
    content: "🛰️";
    margin-right: 0.5rem;
    animation: orbit 4s linear infinite;
}

@keyframes orbit {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.system-status {
    display: flex;
    align-items: center;
    gap: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
}

.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    animation: pulse 2s infinite;
    margin-right: 0.3rem;
}

.status-online { background-color: #28a745; box-shadow: 0 0 8px #28a745; }
.status-warning { background-color: #ffc107; box-shadow: 0 0 8px #ffc107; }
.status-offline { background-color: #dc3545; box-shadow: 0 0 8px #dc3545; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

.utc-clock {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    color: #00c9ff;
    font-weight: 500;
}

/* Main Content Spacing */
.main-content {
    margin-top: 80px;
    padding: 1rem;
}

/* Panel Styling */
.genesis-panel {
    background: rgba(16, 22, 58, 0.8);
    border: 1px solid rgba(0, 201, 255, 0.2);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(5px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.genesis-panel h3 {
    color: #00c9ff;
    font-family: 'Roboto Mono', monospace;
    font-weight: 500;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(0, 201, 255, 0.3);
    padding-bottom: 0.5rem;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(0, 201, 255, 0.1) 0%, rgba(16, 22, 58, 0.8) 100%);
    border: 1px solid rgba(0, 201, 255, 0.3);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
    margin-bottom: 1rem;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 201, 255, 0.2);
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00c9ff;
    text-shadow: 0 0 5px rgba(0, 201, 255, 0.5);
}

.metric-label {
    font-size: 0.85rem;
    color: #a0a9c0;
    margin-top: 0.3rem;
}

/* Satellite Tracker */
.satellite-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.satellite-badge {
    background: rgba(0, 201, 255, 0.1);
    border: 1px solid rgba(0, 201, 255, 0.3);
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.satellite-badge::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0, 201, 255, 0.2), transparent);
    transition: left 0.5s;
}

.satellite-badge:hover::before {
    left: 100%;
}

.satellite-id {
    font-family: 'Roboto Mono', monospace;
    font-weight: 500;
    color: #00c9ff;
    font-size: 0.9rem;
}

.satellite-error {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #e6edf3;
    margin-top: 0.3rem;
}

/* Alert Feed Styling */
.alert-feed {
    background: rgba(10, 15, 35, 0.9);
    border: 1px solid rgba(0, 201, 255, 0.2);
    border-radius: 8px;
    padding: 1rem;
    height: 200px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

.alert-item {
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateX(-10px); }
    to { opacity: 1; transform: translateX(0); }
}

.alert-critical { color: #dc3545; }
.alert-warning { color: #ffc107; }
.alert-info { color: #00c9ff; }
.alert-success { color: #28a745; }

/* Enhanced Streamlit Components */
.stSidebar {
    background: linear-gradient(180deg, #161b2e 0%, #1a1f3a 100%);
    border-right: 1px solid rgba(0, 201, 255, 0.2);
}

.stButton > button {
    background: linear-gradient(135deg, #00c9ff 0%, #0099cc 100%);
    color: #0b0f23;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 201, 255, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 201, 255, 0.4);
}

.stSelectbox > div > div {
    background-color: rgba(16, 22, 58, 0.8);
    border: 1px solid rgba(0, 201, 255, 0.3);
    color: #e6edf3;
}

.stSlider > div > div > div > div {
    background-color: #00c9ff;
}

h1, h2, h3 {
    color: #00c9ff;
    font-family: 'Roboto Mono', monospace;
}

.stMarkdown {
    color: #e6edf3;
}

/* Responsive Design */
@media (max-width: 768px) {
    .genesis-topbar {
        padding: 0.5rem 1rem;
        height: 50px;
    }
    
    .genesis-logo {
        font-size: 1.2rem;
    }
    
    .main-content {
        margin-top: 70px;
        padding: 0.5rem;
    }
    
    .satellite-grid {
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 0.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Real-time Top Navigation Bar
def render_topbar():
    st.markdown("""
    <div class="genesis-topbar">
        <div class="genesis-logo">
            GENESIS-AI <span style="font-weight:300;font-size:0.8rem;margin-left:0.5rem;">v1.0</span>
        </div>
        <div class="system-status">
            <div style="display:flex;align-items:center;">
                <div class="status-indicator status-online"></div>
                <span>OPERATIONAL</span>
            </div>
            <div class="utc-clock" id="utc-clock">Loading...</div>
        </div>
    </div>
    
    <script>
    function updateUTCClock() {
        const now = new Date();
        const utcString = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
        const clockElement = document.getElementById('utc-clock');
        if (clockElement) {
            clockElement.textContent = utcString;
        }
    }
    
    // Update immediately and then every second
    updateUTCClock();
    setInterval(updateUTCClock, 1000);
    </script>
    """, unsafe_allow_html=True)

render_topbar()

# Add spacing for fixed topbar
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Enhanced UI Components
def render_satellite_tracker(predictions_df):
    st.markdown('<div class="genesis-panel">', unsafe_allow_html=True)
    st.markdown("### 🛰️ Live Satellite Status")
    
    if predictions_df is not None and not predictions_df.empty:
        # Group by satellite
        sat_summary = predictions_df.groupby('sat_id').agg({
            'predicted_error': ['mean', 'std'],
            'orbit_class': 'first'
        }).round(4)
        
        sat_summary.columns = ['mean_error', 'std_error', 'orbit_class']
        sat_summary = sat_summary.reset_index()
        
        # Create satellite grid
        st.markdown('<div class="satellite-grid">', unsafe_allow_html=True)
        
        for _, sat in sat_summary.iterrows():
            error_level = "normal"
            if abs(sat['mean_error']) > 0.1:
                error_level = "warning"
            if abs(sat['mean_error']) > 0.5:
                error_level = "critical"
            
            st.markdown(f"""
            <div class="satellite-badge">
                <div class="satellite-id">{sat['sat_id']}</div>
                <div class="satellite-error">{sat['mean_error']:.3f} ±{sat['std_error']:.3f}</div>
                <div style="font-size:0.7rem;color:#a0a9c0;margin-top:0.2rem;">{sat['orbit_class']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;color:#a0a9c0;padding:2rem;">No satellite data available</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_enhanced_metrics(col1, col2, col3, col4, metrics):
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('satellites', 0)}</div>
            <div class="metric-label">Active Satellites</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('orbit_classes', 0)}</div>
            <div class="metric-label">Orbit Classes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('time_span', 0)}d</div>
            <div class="metric-label">Data Span</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('records', 0)}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)

def render_health_gauge(current_rmse, max_range=2.0):
    # Determine gauge color based on RMSE
    if current_rmse < 0.5:
        color = "#28a745"  # Green
    elif current_rmse < 1.0:
        color = "#ffc107"  # Yellow
    else:
        color = "#dc3545"  # Red
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_rmse,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "System RMSE (ns)", 'font': {'color': '#e6edf3', 'size': 16}},
        delta={'reference': 0.5, 'increasing': {'color': "#dc3545"}, 'decreasing': {'color': "#28a745"}},
        gauge={
            'axis': {'range': [None, max_range], 'tickcolor': '#e6edf3'},
            'bar': {'color': color},
            'bgcolor': 'rgba(16, 22, 58, 0.8)',
            'borderwidth': 2,
            'bordercolor': 'rgba(0, 201, 255, 0.3)',
            'steps': [
                {'range': [0, 0.5], 'color': 'rgba(40, 167, 69, 0.3)'},
                {'range': [0.5, 1.0], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [1.0, max_range], 'color': 'rgba(220, 53, 69, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#00c9ff", 'width': 4},
                'thickness': 0.75,
                'value': 1.0
            }
        }
    ))
    
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e6edf3', 'family': 'JetBrains Mono'},
        height=300
    )
    
    return fig_gauge

def render_alert_feed():
    st.markdown('<div class="genesis-panel">', unsafe_allow_html=True)
    st.markdown("### 🚨 Mission Alerts")
    
    try:
        from genesis_ai.monitor.alerts import AlertManager
        alert_manager = AlertManager()
        recent_alerts = alert_manager.get_recent_alerts(hours=24)
        
        st.markdown('<div class="alert-feed">', unsafe_allow_html=True)
        
        if recent_alerts:
            for alert in recent_alerts[-10:]:  # Show last 10 alerts
                severity_class = f"alert-{alert.severity}"
                timestamp = alert.created_at.strftime('%H:%M:%S')
                
                st.markdown(f"""
                <div class="alert-item {severity_class}">
                    <strong>[{timestamp}]</strong> {alert.alert_type.upper()}: {alert.message[:60]}{'...' if len(alert.message) > 60 else ''}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-item alert-success">All systems nominal - no recent alerts</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f'<div class="alert-item alert-warning">Alert system offline: {str(e)[:50]}...</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.title("🛰️ GENESIS-AI Mission Control")
st.caption("Operational GNSS Error Forecasting Platform — Integrated for ISRO")
st.divider()

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    model_type = st.selectbox("Model Engine", ["gru", "transformer"])
    seq_len = st.slider("Sequence Length", 8, 24, 12)
    hidden_size = st.slider("Hidden Units", 32, 256, 128, 32)
    api_url = st.text_input("Inference API URL", "http://127.0.0.1:8000")
    auto_refresh = st.checkbox("Auto Refresh (every 60s)", False)
    uploaded = st.file_uploader("Upload GNSS Error Data (CSV)", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
    # API Health Check
    try:
        client = GenesisClient(base_url=api_url)
        health = client.health_check()
        st.success("✅ API Service Online")
        st.json(health)
    except Exception as e:
        st.error(f"❌ API Service Offline: {str(e)[:50]}...")

# Space Weather & Environmental Conditions Panel
st.subheader("🌦️ Space Weather & Environmental Conditions")
with st.container():
    try:
        # Get comprehensive environmental status
        env_monitor = EnvironmentalMonitor()
        env_status = env_monitor.get_comprehensive_status()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Space Weather
        sw_data = env_status["space_weather"]
        if "error" not in sw_data:
            kp_val = sw_data.get("kp_index", 0)
            severity = sw_data.get("severity", "unknown")
            col1.metric(
                "Kp Index", 
                f"{kp_val:.1f}",
                delta=severity.title(),
                help="Geomagnetic disturbance level (0–9). Higher values indicate more ionospheric disturbance."
            )
        else:
            col1.error("Space Weather Offline")
        
        # Ionosphere
        ion_data = env_status["ionosphere"]
        if "error" not in ion_data:
            tec_val = ion_data.get("TEC_avg", 0)
            quality = ion_data.get("quality", "unknown")
            col2.metric(
                "Avg TEC", 
                f"{tec_val:.1f} TECU",
                delta=quality.title(),
                help="Total Electron Content in TEC Units. Affects GNSS signal delay."
            )
        else:
            col2.error("Ionosphere Data Offline")
        
        # Solar Activity
        solar_data = env_status["solar_activity"]
        if "error" not in solar_data:
            f107_val = solar_data.get("f107_flux", 0)
            activity = solar_data.get("activity_level", "unknown")
            col3.metric(
                "F10.7 Flux",
                f"{f107_val:.1f} SFU",
                delta=activity.replace("_", " ").title(),
                help="Solar radio flux at 10.7 cm. Indicates solar activity level."
            )
        else:
            col3.error("Solar Data Offline")
        
        # Overall Health
        health = env_status["overall_health"]
        health_color = {
            "excellent": "🟢",
            "good": "🟡", 
            "fair": "🟠",
            "poor": "🔴"
        }.get(health["status"], "⚪")
        
        col4.metric(
            "GNSS Environment",
            f"{health_color} {health['score']:.0f}/100",
            delta=health["status"].title(),
            help="Overall GNSS operating environment health score"
        )
        
        # Environmental recommendations
        if env_status["recommendations"]:
            with st.expander("📋 Operational Recommendations"):
                for rec in env_status["recommendations"]:
                    st.write(f"• {rec}")
        
        # Last update time
        st.caption(f"Environmental data updated: {datetime.now().strftime('%H:%M:%S UTC')}")
        
    except Exception as e:
        st.error(f"Environmental monitoring error: {str(e)[:100]}...")

st.divider()

if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["timestamp"])
    
    # Enhanced Data Summary
    col1, col2, col3, col4 = st.columns(4)
    metrics_data = {
        'records': len(df),
        'satellites': df['sat_id'].nunique(),
        'orbit_classes': df['orbit_class'].nunique(),
        'time_span': (df['timestamp'].max() - df['timestamp'].min()).days
    }
    render_enhanced_metrics(col1, col2, col3, col4, metrics_data)
    
    st.success(f"📡 Loaded telemetry from {df['sat_id'].nunique()} satellites across {df['orbit_class'].nunique()} orbit classes")

    if st.button("🚀 Run ISRO Forecast", type="primary"):
        with st.spinner("Computing multi-horizon forecasts via /predict_pro..."):
            try:
                client = GenesisClient(base_url=api_url)
                result = client.predict_pro(df, model_type=model_type, seq_len=seq_len, hidden_size=hidden_size)
                preds = pd.DataFrame(result["predictions"])
                preds["timestamp"] = pd.to_datetime(preds["timestamp"])
                
                st.success("🎯 Forecast computation complete!")
                
                # Store in session state for auto-refresh
                st.session_state.predictions = preds
                st.session_state.forecast_info = result["info"]
                st.session_state.last_update = datetime.now()
                
            except Exception as e:
                st.error(f"❌ Forecast failed: {str(e)}")
                st.stop()

    # Display results if available
    if "predictions" in st.session_state:
        preds = st.session_state.predictions
        info = st.session_state.forecast_info
        
        st.subheader("📊 Multi-Horizon Forecast Results")
        
        # Enhanced Forecast Info Panel
        col1, col2, col3, col4 = st.columns(4)
        forecast_metrics = {
            'model_type': info.get("model_type", "N/A").upper(),
            'seq_len': info.get("seq_len", "N/A"),
            'hidden_size': info.get("hidden_size", "N/A"),
            'confidence': info.get("gp_confidence", "N/A")
        }
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{forecast_metrics['model_type']}</div>
                <div class="metric-label">Model Type</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{forecast_metrics['seq_len']}</div>
                <div class="metric-label">Sequence Length</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{forecast_metrics['hidden_size']}</div>
                <div class="metric-label">Hidden Size</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{forecast_metrics['confidence']}</div>
                <div class="metric-label">GP Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Live Satellite Tracker
        render_satellite_tracker(preds)
        
        # Results Table
        st.dataframe(
            preds.style.background_gradient(subset=["predicted_error"], cmap="RdYlBu_r")
                      .format({"predicted_error": "{:.4f}", "lower_bound": "{:.4f}", "upper_bound": "{:.4f}"}),
            width='stretch'
        )

        # Plotting section
        st.subheader("🔮 Predicted Error Timeline")
        
        # Create interactive plot with confidence bands
        fig = go.Figure()
        
        # Group by satellite for different colors
        satellites = preds['sat_id'].unique()
        # Use ISRO-themed colors
        isro_colors = ["#00c9ff", "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57"]

        for i, sat in enumerate(satellites):
            sat_data = preds[preds['sat_id'] == sat].sort_values('timestamp')
            color = isro_colors[i % len(isro_colors)]

            # Main prediction line
            fig.add_trace(go.Scatter(
                x=sat_data["timestamp"],
                y=sat_data["predicted_error"],
                mode="lines+markers",
                name=f"{sat} - Predicted",
                line=dict(color=color, width=3),
                marker=dict(size=8)
            ))

            # Confidence band
            if "lower_bound" in sat_data.columns and "upper_bound" in sat_data.columns:
                # Convert hex color to rgba
                hex_color = color.lstrip('#')
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                
                fig.add_trace(go.Scatter(
                    x=pd.concat([sat_data["timestamp"], sat_data["timestamp"][::-1]]),
                    y=pd.concat([sat_data["upper_bound"], sat_data["lower_bound"][::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({r}, {g}, {b}, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    name=f"{sat} - Confidence",
                    hovertemplate="Confidence Band<extra></extra>"
                ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f23",
            plot_bgcolor="#161b2e",
            font=dict(color="#e6edf3", size=12),
            xaxis_title="Time (UTC)",
            yaxis_title="Predicted Error (ns or m)",
            height=600,
            hovermode="x unified",
            legend=dict(
                bgcolor="rgba(22, 27, 46, 0.8)",
                bordercolor="#00c9ff",
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # System Health Gauge
        col1, col2 = st.columns([2, 1])
        with col2:
            # Calculate current RMSE from predictions
            current_rmse = preds['predicted_error'].std() if not preds.empty else 0.5
            health_gauge = render_health_gauge(current_rmse)
            st.plotly_chart(health_gauge, width='stretch')
        
        with col1:
            # Alert Feed
            render_alert_feed()

        # ISRO Ground Station Network Map
        st.subheader("🗺️ ISRO Ground Station Network")
        
        try:
            stations_df = get_ground_stations()
            
            # Create ground station map using Plotly
            fig_geo = go.Figure()
            
            # Add ground stations
            online_stations = stations_df[stations_df['status'] == 'online']
            offline_stations = stations_df[stations_df['status'] != 'online']
            
            if len(online_stations) > 0:
                fig_geo.add_trace(go.Scattergeo(
                    lon=online_stations['lon'],
                    lat=online_stations['lat'],
                    text=online_stations['station'] + '<br>TEC: ' + online_stations['TEC'].astype(str) + ' TECU',
                    mode='markers+text',
                    marker=dict(size=12, color='#00c9ff', symbol='circle'),
                    textposition="top center",
                    name='Online Stations',
                    hovertemplate='<b>%{text}</b><br>Status: Online<extra></extra>'
                ))
            
            if len(offline_stations) > 0:
                fig_geo.add_trace(go.Scattergeo(
                    lon=offline_stations['lon'],
                    lat=offline_stations['lat'],
                    text=offline_stations['station'],
                    mode='markers+text',
                    marker=dict(size=10, color='#ff6b6b', symbol='x'),
                    textposition="top center",
                    name='Offline Stations',
                    hovertemplate='<b>%{text}</b><br>Status: Maintenance<extra></extra>'
                ))
            
            fig_geo.update_layout(
                geo=dict(
                    projection_type='natural earth',
                    showland=True,
                    landcolor='#1e2130',
                    showocean=True,
                    oceancolor='#0b0f23',
                    showlakes=True,
                    lakecolor='#0b0f23',
                    showrivers=True,
                    rivercolor='#0b0f23',
                    showcountries=True,
                    countrycolor='#404040',
                    center=dict(lat=20, lon=78),  # Center on India
                    projection_scale=3
                ),
                title=dict(
                    text="ISRO Ionospheric Monitoring Network",
                    font=dict(color='#e6edf3', size=16)
                ),
                paper_bgcolor='#0b0f23',
                plot_bgcolor='#0b0f23',
                font=dict(color='#e6edf3'),
                height=400,
                showlegend=True,
                legend=dict(
                    bgcolor="rgba(22, 27, 46, 0.8)",
                    bordercolor="#00c9ff",
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig_geo, width='stretch')
            
            # Station status summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Online Stations", len(online_stations))
            with col2:
                st.metric("Maintenance", len(offline_stations))
            with col3:
                avg_tec = stations_df['TEC'].mean()
                st.metric("Network Avg TEC", f"{avg_tec:.1f} TECU")
                
        except Exception as e:
            st.error(f"Ground station map error: {str(e)[:50]}...")

        # Satellite Summary Dashboard
        st.subheader("📡 Satellite Performance Dashboard")
        
        sat_summary = preds.groupby(["sat_id", "orbit_class"]).agg({
            "predicted_error": ["mean", "std", "min", "max"],
            "lower_bound": "min",
            "upper_bound": "max"
        }).round(4)
        
        sat_summary.columns = ["Mean Error", "Std Error", "Min Error", "Max Error", "Min Bound", "Max Bound"]
        sat_summary = sat_summary.reset_index()
        
        # Color-code by orbit class
        orbit_colors = {"GEO/GSO": "#ff6b6b", "MEO": "#4ecdc4", "LEO": "#45b7d1"}
        
        for orbit in sat_summary["orbit_class"].unique():
            orbit_data = sat_summary[sat_summary["orbit_class"] == orbit]
            st.markdown(f"#### {orbit} Satellites")
            
            cols = st.columns(len(orbit_data))
            for i, (_, row) in enumerate(orbit_data.iterrows()):
                with cols[i % len(cols)]:
                    st.metric(
                        label=f"🛰️ {row['sat_id']}",
                        value=f"{row['Mean Error']:.4f}",
                        delta=f"±{row['Std Error']:.4f}"
                    )
        
        # Last update info
        if "last_update" in st.session_state:
            st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S UTC')}")

else:
    st.info("📂 Upload GNSS CSV data to begin operational forecasting.")
    
    # Show sample data format
    with st.expander("📋 Expected Data Format"):
        st.markdown("""
        Your CSV should contain these columns:
        - `sat_id`: Satellite identifier (e.g., "SAT-001")
        - `orbit_class`: "GEO/GSO", "MEO", or "LEO"
        - `quantity`: "clock" or "ephem"
        - `timestamp`: ISO format datetime (e.g., "2025-01-01T00:00:00Z")
        - `error`: Numerical error value (ns for clock, m for ephemeris)
        """)

# Operations Console
st.markdown("## 🧭 Operations Console")
tab1, tab2, tab3 = st.tabs(["Database Records", "Alerts & Monitoring", "System Health"])

with tab1:
    st.markdown("### 📊 Database Statistics")
    
    try:
        from genesis_ai.db.models import ForecastRecord, TrainingRun, AlertRecord, get_engine, get_session, get_db_stats, init_database
        
        # Initialize database if needed
        engine = init_database()
        stats = get_db_stats(engine)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Forecast Records", stats.get('forecast_records', 0))
        with col2:
            st.metric("Training Runs", stats.get('training_runs', 0))
        with col3:
            st.metric("Alert Records", stats.get('alert_records', 0))
        
        # Show recent records
        if st.button("📋 Show Recent Forecasts"):
            session = get_session(engine)
            try:
                records = session.query(ForecastRecord).order_by(ForecastRecord.created_at.desc()).limit(20).all()
                if records:
                    df_records = pd.DataFrame([{
                        'ID': r.id,
                        'Satellite': r.sat_id,
                        'Orbit': r.orbit_class,
                        'Quantity': r.quantity,
                        'Predicted Error': f"{r.predicted_error:.4f}",
                        'Confidence': f"[{r.lower_bound:.3f}, {r.upper_bound:.3f}]" if r.lower_bound else "N/A",
                        'Model': r.model_type,
                        'Created': r.created_at.strftime('%Y-%m-%d %H:%M')
                    } for r in records])
                    st.dataframe(df_records, width='stretch')
                else:
                    st.info("No forecast records found")
            finally:
                session.close()
        
        # Show recent training runs
        if st.button("🏃 Show Recent Training"):
            session = get_session(engine)
            try:
                runs = session.query(TrainingRun).order_by(TrainingRun.started_at.desc()).limit(10).all()
                if runs:
                    df_runs = pd.DataFrame([{
                        'ID': r.id,
                        'Model': r.model_type,
                        'Status': r.status,
                        'Epochs': r.epochs,
                        'Duration': f"{r.training_duration:.1f}s" if r.training_duration else "N/A",
                        'Final Loss': f"{r.final_loss:.4f}" if r.final_loss else "N/A",
                        'Started': r.started_at.strftime('%Y-%m-%d %H:%M')
                    } for r in runs])
                    st.dataframe(df_runs, width='stretch')
                else:
                    st.info("No training runs found")
            finally:
                session.close()
                
    except Exception as e:
        st.error(f"Database error: {e}")

with tab2:
    st.markdown("### 🚨 Alert System")
    
    try:
        from genesis_ai.monitor.alerts import AlertManager, MonitoringService
        
        alert_manager = AlertManager()
        
        # Alert statistics
        recent_alerts = alert_manager.get_recent_alerts(hours=24)
        critical_alerts = [a for a in recent_alerts if a.severity == 'critical']
        high_alerts = [a for a in recent_alerts if a.severity == 'high']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("24h Alerts", len(recent_alerts))
        with col2:
            st.metric("Critical", len(critical_alerts), delta=None if len(critical_alerts) == 0 else f"+{len(critical_alerts)}")
        with col3:
            st.metric("High Priority", len(high_alerts))
        with col4:
            unack_alerts = [a for a in recent_alerts if not a.acknowledged]
            st.metric("Unacknowledged", len(unack_alerts))
        
        # Recent alerts table
        if recent_alerts:
            st.markdown("#### Recent Alerts")
            df_alerts = pd.DataFrame([{
                'ID': a.id,
                'Type': a.alert_type,
                'Severity': a.severity,
                'Satellite': a.sat_id or 'System',
                'Message': a.message[:50] + '...' if len(a.message) > 50 else a.message,
                'Status': '✅ Ack' if a.acknowledged else '⏳ Pending',
                'Time': a.created_at.strftime('%H:%M')
            } for a in recent_alerts[:10]])
            
            # Color code by severity
            def color_severity(val):
                colors = {
                    'critical': 'background-color: #ffebee',
                    'high': 'background-color: #fff3e0', 
                    'medium': 'background-color: #f3e5f5',
                    'low': 'background-color: #e8f5e8'
                }
                return colors.get(val, '')
            
            styled_df = df_alerts.style.map(color_severity, subset=['Severity'])
            st.dataframe(styled_df, width='stretch')
        else:
            st.success("🎉 No recent alerts - all systems nominal")
        
        # Test alert system
        if st.button("🧪 Test Alert System"):
            alert_id = alert_manager.create_alert(
                alert_type='test',
                severity='low',
                message='Test alert from mission control dashboard',
                details={'source': 'streamlit_ui', 'timestamp': datetime.now().isoformat()}
            )
            st.success(f"Test alert created (ID: {alert_id})")
            
    except Exception as e:
        st.error(f"Alert system error: {e}")

with tab3:
    st.markdown("### 🔧 System Health")
    
    # System status indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Service Status")
        
        # API Health
        try:
            client = GenesisClient(base_url=api_url)
            health = client.health_check()
            st.success("✅ Inference API Online")
        except:
            st.error("❌ Inference API Offline")
        
        # Database Health
        try:
            from sqlalchemy import text
            engine = get_engine()
            session = get_session(engine)
            session.execute(text("SELECT 1"))
            session.close()
            st.success("✅ Database Connected")
        except:
            st.error("❌ Database Offline")
        
        # Data Directory
        data_dir = Path("data")
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            st.success(f"✅ Data Directory ({len(csv_files)} files)")
        else:
            st.warning("⚠️ Data Directory Not Found")
    
    with col2:
        st.markdown("#### Performance Metrics")
        
        # Last update time
        if "last_update" in st.session_state:
            last_update = st.session_state.last_update
            time_since = datetime.now() - last_update
            st.metric("Last Forecast", f"{time_since.seconds//60}m ago")
        
        # System uptime (placeholder)
        st.metric("System Uptime", "24h 15m")
        
        # Memory usage (placeholder)
        st.metric("Memory Usage", "2.1 GB")
        
        # Active connections (placeholder)
        st.metric("Active Sessions", "3")
    
    # System logs (placeholder)
    st.markdown("#### Recent System Events")
    with st.expander("View System Logs"):
        st.code("""
[2025-01-07 17:45:12] INFO - Forecast computation completed for IRNSS-1A
[2025-01-07 17:44:58] INFO - API health check passed
[2025-01-07 17:44:45] INFO - Database connection established
[2025-01-07 17:44:30] INFO - Streamlit dashboard started
[2025-01-07 17:44:15] INFO - GENESIS-AI system initialized
        """)

# Auto-refresh functionality
if auto_refresh and "predictions" in st.session_state:
    time.sleep(1)  # Small delay to prevent excessive refreshing
    st.rerun()

st.markdown("---")
st.caption("© 2025 GENESIS-AI | ISRO GNSS Research Division — Real-time Predictive Analytics")
