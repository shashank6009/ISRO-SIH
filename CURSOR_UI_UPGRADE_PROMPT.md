# 🎨 GENESIS-AI ISRO Mission Control UI Upgrade

## Objective
Transform GENESIS-AI into a cinematic, presentation-grade ISRO mission control dashboard with professional visual identity, real-time elements, and operational polish while maintaining all existing functionality.

## Visual Identity Specification

**Color Palette:**
- Primary Background: `#0b0f23` (Deep Navy)
- Accent/Highlights: `#00c9ff` (Glowing Cyan) 
- Typography: `#e6edf3` (Clean White)
- Panel Background: `#10163a` (Darker Navy)
- Success: `#28a745`, Warning: `#ffc107`, Critical: `#dc3545`

**Typography:**
- Headers: `'Roboto Mono', monospace` (Technical precision)
- Body: `'Inter', sans-serif` (Clean readability)
- Metrics: `'JetBrains Mono', monospace` (Data display)

## Implementation Instructions

### 1. Create Enhanced CSS Theme

Replace the existing CSS in `src/genesis_ai/app/main.py` with this comprehensive theme:

```python
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

/* Progress Animations */
.genesis-progress {
    background: rgba(0, 201, 255, 0.2);
    border-radius: 10px;
    overflow: hidden;
    height: 6px;
    margin: 1rem 0;
}

.genesis-progress-bar {
    background: linear-gradient(90deg, #00c9ff, #0099cc);
    height: 100%;
    border-radius: 10px;
    animation: progressGlow 2s ease-in-out infinite alternate;
}

@keyframes progressGlow {
    0% { box-shadow: 0 0 5px rgba(0, 201, 255, 0.5); }
    100% { box-shadow: 0 0 20px rgba(0, 201, 255, 0.8); }
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

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(16, 22, 58, 0.5);
}

::-webkit-scrollbar-thumb {
    background: rgba(0, 201, 255, 0.5);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 201, 255, 0.8);
}
</style>
""", unsafe_allow_html=True)
```

### 2. Add Persistent Top Navigation Bar

Add this immediately after the CSS:

```python
# Real-time UTC Clock Component
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
```

### 3. Create Real-time Satellite Tracker Component

Add this new function before the main dashboard content:

```python
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
```

### 4. Add Health Gauge Component

```python
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
```

### 5. Create Live Alert Feed

```python
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
```

### 6. Enhanced Metrics Display

Replace existing metric displays with:

```python
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
```

### 7. Integration Instructions

1. **Replace the existing CSS** in `main.py` with the comprehensive theme above
2. **Add the topbar function** right after the CSS
3. **Insert satellite tracker** in the main dashboard area
4. **Add health gauge** below the main forecast plots
5. **Include alert feed** in the right sidebar
6. **Update metrics display** to use the enhanced cards
7. **Test responsiveness** on different screen sizes

### 8. Final Polish

- Add smooth loading animations during API calls
- Implement hover effects on all interactive elements
- Ensure all plots use the consistent color scheme
- Add subtle sound effects for alerts (optional)
- Test in both light and dark environments

This upgrade will transform GENESIS-AI into a cinematic, professional ISRO mission control interface while maintaining all existing functionality and technical excellence.
