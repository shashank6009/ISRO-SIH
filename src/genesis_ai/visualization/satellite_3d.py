"""
3D Satellite Constellation Viewer for GENESIS-AI
Interactive 3D visualization of satellite positions and error predictions
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import math

class Satellite3DViewer:
    """3D visualization of satellite constellation with error predictions."""
    
    def __init__(self):
        # Earth parameters
        self.earth_radius = 6371  # km
        
        # Orbital parameters for different classes
        self.orbital_params = {
            "LEO": {"altitude": 800, "period": 100, "inclination": 98},
            "MEO": {"altitude": 20200, "period": 720, "inclination": 55},
            "GEO/GSO": {"altitude": 35786, "period": 1440, "inclination": 0}
        }
        
        # Color scheme for different error levels
        self.error_colors = {
            "low": "#28a745",      # Green
            "medium": "#ffc107",   # Yellow  
            "high": "#fd7e14",     # Orange
            "critical": "#dc3545"  # Red
        }
        
    def create_3d_constellation(self, satellite_data: pd.DataFrame, 
                               current_time: datetime = None) -> go.Figure:
        """
        Create interactive 3D satellite constellation visualization.
        
        Args:
            satellite_data: DataFrame with satellite positions and errors
            current_time: Current time for orbital calculations
            
        Returns:
            Plotly 3D figure
        """
        if current_time is None:
            current_time = datetime.now()
            
        fig = go.Figure()
        
        # Add Earth sphere
        self._add_earth_sphere(fig)
        
        # Add satellites
        self._add_satellites(fig, satellite_data, current_time)
        
        # Add orbital paths
        self._add_orbital_paths(fig, satellite_data)
        
        # Add ground tracks
        self._add_ground_tracks(fig, satellite_data)
        
        # Configure layout
        self._configure_3d_layout(fig)
        
        return fig
        
    def _add_earth_sphere(self, fig: go.Figure):
        """Add Earth sphere to the 3D plot."""
        # Create sphere coordinates
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        
        x = self.earth_radius * np.outer(np.cos(u), np.sin(v))
        y = self.earth_radius * np.outer(np.sin(u), np.sin(v))
        z = self.earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Add Earth surface
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, '#1e3d59'], [0.5, '#2e5266'], [1, '#3e6b73']],
            showscale=False,
            opacity=0.8,
            name="Earth",
            hoverinfo='skip'
        ))
        
    def _add_satellites(self, fig: go.Figure, satellite_data: pd.DataFrame, 
                       current_time: datetime):
        """Add satellite positions with error-based coloring."""
        for _, sat in satellite_data.iterrows():
            # Calculate 3D position
            pos = self._calculate_satellite_position(sat, current_time)
            
            # Determine error level and color
            error_level = self._categorize_error(sat.get('error', 0))
            color = self.error_colors[error_level]
            
            # Size based on error magnitude
            size = max(8, min(20, abs(sat.get('error', 0)) * 200))
            
            # Add satellite marker
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                    symbol='diamond',
                    line=dict(width=2, color='white')
                ),
                text=[sat['sat_id']],
                textposition="top center",
                name=f"{sat['sat_id']} ({error_level})",
                hovertemplate=(
                    f"<b>{sat['sat_id']}</b><br>"
                    f"Orbit: {sat.get('orbit_class', 'Unknown')}<br>"
                    f"Error: {sat.get('error', 0):.4f} ns/m<br>"
                    f"Altitude: {self.orbital_params.get(sat.get('orbit_class', 'LEO'), {}).get('altitude', 0)} km<br>"
                    f"Status: {error_level.title()}<br>"
                    "<extra></extra>"
                )
            ))
            
    def _calculate_satellite_position(self, satellite: pd.Series, 
                                    current_time: datetime) -> Tuple[float, float, float]:
        """Calculate 3D position of satellite based on orbital parameters."""
        orbit_class = satellite.get('orbit_class', 'LEO')
        params = self.orbital_params.get(orbit_class, self.orbital_params['LEO'])
        
        # Get orbital elements
        altitude = params['altitude']
        period_min = params['period']
        inclination = math.radians(params['inclination'])
        
        # Calculate orbital radius
        radius = self.earth_radius + altitude
        
        # Time-based orbital position (simplified)
        time_fraction = (current_time.minute + current_time.second/60) / period_min
        mean_anomaly = 2 * math.pi * time_fraction
        
        # Add satellite-specific phase offset
        sat_id_hash = hash(satellite.get('sat_id', '')) % 360
        phase_offset = math.radians(sat_id_hash)
        
        # Calculate position in orbital plane
        true_anomaly = mean_anomaly + phase_offset
        
        # Position in orbital coordinate system
        x_orbit = radius * math.cos(true_anomaly)
        y_orbit = radius * math.sin(true_anomaly) * math.cos(inclination)
        z_orbit = radius * math.sin(true_anomaly) * math.sin(inclination)
        
        return (x_orbit, y_orbit, z_orbit)
        
    def _categorize_error(self, error: float) -> str:
        """Categorize error level for coloring."""
        abs_error = abs(error)
        
        if abs_error < 0.02:
            return "low"
        elif abs_error < 0.05:
            return "medium"
        elif abs_error < 0.1:
            return "high"
        else:
            return "critical"
            
    def _add_orbital_paths(self, fig: go.Figure, satellite_data: pd.DataFrame):
        """Add orbital path traces for each satellite."""
        for _, sat in satellite_data.iterrows():
            orbit_class = sat.get('orbit_class', 'LEO')
            params = self.orbital_params.get(orbit_class, self.orbital_params['LEO'])
            
            # Generate orbital path points
            path_points = self._generate_orbital_path(params)
            
            # Add orbital path
            fig.add_trace(go.Scatter3d(
                x=path_points[0], y=path_points[1], z=path_points[2],
                mode='lines',
                line=dict(
                    color=self.error_colors["medium"],
                    width=2,
                    dash='dot'
                ),
                opacity=0.3,
                name=f"{orbit_class} Orbit",
                showlegend=False,
                hoverinfo='skip'
            ))
            
    def _generate_orbital_path(self, params: Dict[str, float]) -> Tuple[List, List, List]:
        """Generate orbital path coordinates."""
        altitude = params['altitude']
        inclination = math.radians(params['inclination'])
        radius = self.earth_radius + altitude
        
        # Generate points along orbit
        angles = np.linspace(0, 2*math.pi, 100)
        
        x_points = []
        y_points = []
        z_points = []
        
        for angle in angles:
            x = radius * math.cos(angle)
            y = radius * math.sin(angle) * math.cos(inclination)
            z = radius * math.sin(angle) * math.sin(inclination)
            
            x_points.append(x)
            y_points.append(y)
            z_points.append(z)
            
        return (x_points, y_points, z_points)
        
    def _add_ground_tracks(self, fig: go.Figure, satellite_data: pd.DataFrame):
        """Add ground track projections on Earth surface."""
        for _, sat in satellite_data.iterrows():
            # Generate ground track points (simplified)
            track_points = self._generate_ground_track(sat)
            
            if track_points:
                fig.add_trace(go.Scatter3d(
                    x=track_points[0], y=track_points[1], z=track_points[2],
                    mode='lines',
                    line=dict(
                        color='cyan',
                        width=3
                    ),
                    opacity=0.6,
                    name=f"{sat['sat_id']} Ground Track",
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
    def _generate_ground_track(self, satellite: pd.Series) -> Optional[Tuple[List, List, List]]:
        """Generate ground track projection on Earth surface."""
        orbit_class = satellite.get('orbit_class', 'LEO')
        
        # Only show ground tracks for LEO satellites (more interesting)
        if orbit_class != 'LEO':
            return None
            
        # Generate simplified ground track
        lats = np.linspace(-60, 60, 50)  # Latitude range
        lons = np.linspace(-180, 180, 50)  # Longitude range
        
        x_points = []
        y_points = []
        z_points = []
        
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            # Convert to Cartesian coordinates on Earth surface
            lat_rad = math.radians(lat)
            lon_rad = math.radians(lon)
            
            x = self.earth_radius * math.cos(lat_rad) * math.cos(lon_rad)
            y = self.earth_radius * math.cos(lat_rad) * math.sin(lon_rad)
            z = self.earth_radius * math.sin(lat_rad)
            
            x_points.append(x)
            y_points.append(y)
            z_points.append(z)
            
        return (x_points, y_points, z_points)
        
    def _configure_3d_layout(self, fig: go.Figure):
        """Configure 3D plot layout and styling."""
        fig.update_layout(
            title={
                'text': "🛰️ GENESIS-AI: Live Satellite Constellation",
                'x': 0.5,
                'font': {'size': 20, 'color': '#00c9ff'}
            },
            scene=dict(
                xaxis=dict(
                    title="X (km)",
                    backgroundcolor="#0b0f23",
                    gridcolor="#2a3f5f",
                    showbackground=True,
                    zerolinecolor="#2a3f5f"
                ),
                yaxis=dict(
                    title="Y (km)",
                    backgroundcolor="#0b0f23", 
                    gridcolor="#2a3f5f",
                    showbackground=True,
                    zerolinecolor="#2a3f5f"
                ),
                zaxis=dict(
                    title="Z (km)",
                    backgroundcolor="#0b0f23",
                    gridcolor="#2a3f5f", 
                    showbackground=True,
                    zerolinecolor="#2a3f5f"
                ),
                bgcolor="#0b0f23",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='cube'
            ),
            paper_bgcolor="#0b0f23",
            plot_bgcolor="#0b0f23",
            font=dict(color="#e6edf3"),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(22, 27, 46, 0.8)",
                bordercolor="#00c9ff",
                borderwidth=1
            ),
            height=700
        )
        
    def create_error_timeline_3d(self, satellite_data: pd.DataFrame, 
                                time_range: int = 24) -> go.Figure:
        """Create 3D timeline showing error evolution over time."""
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'scatter3d'}]],
            subplot_titles=["Error Evolution Timeline"]
        )
        
        # Generate time points
        current_time = datetime.now()
        time_points = [current_time - timedelta(hours=i) for i in range(time_range)]
        
        for _, sat in satellite_data.iterrows():
            # Simulate error history (in practice, get from database)
            errors = np.random.normal(sat.get('error', 0), 0.01, time_range)
            
            # Create 3D trajectory
            x_coords = list(range(time_range))
            y_coords = [hash(sat['sat_id']) % 10] * time_range  # Satellite lane
            z_coords = errors.tolist()
            
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='lines+markers',
                line=dict(width=4, color=self.error_colors[self._categorize_error(sat.get('error', 0))]),
                marker=dict(size=4),
                name=sat['sat_id'],
                hovertemplate=(
                    f"<b>{sat['sat_id']}</b><br>"
                    "Time: %{x}h ago<br>"
                    "Error: %{z:.4f} ns/m<br>"
                    "<extra></extra>"
                )
            ))
            
        fig.update_layout(
            title="🕐 Satellite Error Evolution (24h)",
            scene=dict(
                xaxis_title="Hours Ago",
                yaxis_title="Satellite ID",
                zaxis_title="Error (ns/m)",
                bgcolor="#0b0f23"
            ),
            paper_bgcolor="#0b0f23",
            font=dict(color="#e6edf3"),
            height=600
        )
        
        return fig
        
    def create_coverage_map(self, satellite_data: pd.DataFrame) -> go.Figure:
        """Create global coverage visualization."""
        fig = go.Figure()
        
        # Add world map
        fig.add_trace(go.Scattergeo(
            lon=[],
            lat=[],
            mode='markers'
        ))
        
        # Add satellite coverage circles
        for _, sat in satellite_data.iterrows():
            orbit_class = sat.get('orbit_class', 'LEO')
            params = self.orbital_params.get(orbit_class, self.orbital_params['LEO'])
            
            # Calculate coverage radius (simplified)
            altitude = params['altitude']
            coverage_radius = math.degrees(math.atan(self.earth_radius / (self.earth_radius + altitude)))
            
            # Satellite ground position (simplified)
            lat = sat.get('lat', 0)
            lon = sat.get('lon', 0)
            
            # Add coverage circle
            circle_lats, circle_lons = self._generate_coverage_circle(lat, lon, coverage_radius)
            
            fig.add_trace(go.Scattergeo(
                lon=circle_lons,
                lat=circle_lats,
                mode='lines',
                line=dict(
                    width=2,
                    color=self.error_colors[self._categorize_error(sat.get('error', 0))]
                ),
                name=f"{sat['sat_id']} Coverage",
                hovertemplate=f"<b>{sat['sat_id']}</b><br>Coverage Area<extra></extra>"
            ))
            
            # Add satellite position
            fig.add_trace(go.Scattergeo(
                lon=[lon],
                lat=[lat],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=self.error_colors[self._categorize_error(sat.get('error', 0))],
                    symbol='diamond'
                ),
                text=[sat['sat_id']],
                textposition="top center",
                name=sat['sat_id'],
                hovertemplate=(
                    f"<b>{sat['sat_id']}</b><br>"
                    f"Error: {sat.get('error', 0):.4f} ns/m<br>"
                    f"Orbit: {orbit_class}<br>"
                    "<extra></extra>"
                )
            ))
            
        fig.update_layout(
            title="🌍 Global Satellite Coverage & Error Status",
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='#1e2130',
                showocean=True,
                oceancolor='#0b0f23',
                showlakes=True,
                lakecolor='#0b0f23',
                showcountries=True,
                countrycolor='#404040'
            ),
            paper_bgcolor='#0b0f23',
            font=dict(color='#e6edf3'),
            height=500
        )
        
        return fig
        
    def _generate_coverage_circle(self, center_lat: float, center_lon: float, 
                                 radius_deg: float) -> Tuple[List[float], List[float]]:
        """Generate circle coordinates for satellite coverage area."""
        angles = np.linspace(0, 2*math.pi, 50)
        
        lats = []
        lons = []
        
        for angle in angles:
            lat = center_lat + radius_deg * math.cos(angle)
            lon = center_lon + radius_deg * math.sin(angle) / math.cos(math.radians(center_lat))
            
            # Keep within valid ranges
            lat = max(-90, min(90, lat))
            lon = ((lon + 180) % 360) - 180
            
            lats.append(lat)
            lons.append(lon)
            
        return lats, lons


def create_satellite_viewer() -> Satellite3DViewer:
    """Factory function to create 3D satellite viewer."""
    return Satellite3DViewer()

def generate_demo_satellite_data() -> pd.DataFrame:
    """Generate demo satellite data for visualization."""
    satellites = [
        {"sat_id": "IRNSS-1A", "orbit_class": "GEO/GSO", "error": 0.045, "lat": 55.0, "lon": 83.0},
        {"sat_id": "IRNSS-1B", "orbit_class": "GEO/GSO", "error": 0.032, "lat": 55.0, "lon": 55.0},
        {"sat_id": "IRNSS-1C", "orbit_class": "GEO/GSO", "error": 0.038, "lat": 55.0, "lon": 111.75},
        {"sat_id": "GSAT-30", "orbit_class": "MEO", "error": 0.067, "lat": 20.0, "lon": 78.0},
        {"sat_id": "GSAT-31", "orbit_class": "MEO", "error": 0.054, "lat": -20.0, "lon": 78.0},
        {"sat_id": "CARTOSAT-2", "orbit_class": "LEO", "error": 0.089, "lat": 0.0, "lon": 78.0},
        {"sat_id": "RESOURCESAT-2", "orbit_class": "LEO", "error": 0.123, "lat": 45.0, "lon": 78.0}
    ]
    
    return pd.DataFrame(satellites)
