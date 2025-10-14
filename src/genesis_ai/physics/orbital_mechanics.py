"""
Physics-Informed Components for GNSS Error Prediction
Implements orbital mechanics constraints and satellite dynamics for enhanced accuracy
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class OrbitalParameters:
    """Standard orbital parameters for GNSS satellites"""
    semi_major_axis: float  # meters
    eccentricity: float
    inclination: float  # radians
    longitude_ascending_node: float  # radians
    argument_perigee: float  # radians
    mean_anomaly: float  # radians
    
class OrbitalMechanics:
    """Orbital mechanics calculations for GNSS satellites"""
    
    # Physical constants
    GM_EARTH = 3.986004418e14  # m³/s² - Earth's gravitational parameter
    EARTH_RADIUS = 6.371e6  # meters
    J2 = 1.08262668e-3  # Earth's oblateness coefficient
    
    # Typical orbital parameters for different satellite types
    ORBIT_PARAMS = {
        'GEO/GSO': {
            'altitude': 35786000,  # 35,786 km
            'period': 86400,  # 24 hours
            'inclination': 0.0,
            'eccentricity': 0.0
        },
        'MEO': {
            'altitude': 20200000,  # ~20,200 km (GPS)
            'period': 43200,  # 12 hours
            'inclination': np.pi/3,  # ~55 degrees
            'eccentricity': 0.02
        },
        'LEO': {
            'altitude': 1200000,  # ~1,200 km
            'period': 6600,  # ~110 minutes
            'inclination': np.pi/2,  # 90 degrees
            'eccentricity': 0.001
        }
    }
    
    @staticmethod
    def kepler_period(semi_major_axis: float) -> float:
        """Calculate orbital period using Kepler's third law"""
        return 2 * np.pi * np.sqrt(semi_major_axis**3 / OrbitalMechanics.GM_EARTH)
    
    @staticmethod
    def mean_motion(semi_major_axis: float) -> float:
        """Calculate mean motion (radians per second)"""
        return np.sqrt(OrbitalMechanics.GM_EARTH / semi_major_axis**3)
    
    @staticmethod
    def j2_perturbation(orbit_params: OrbitalParameters) -> Dict[str, float]:
        """Calculate J2 perturbation effects on orbital elements"""
        a = orbit_params.semi_major_axis
        e = orbit_params.eccentricity
        i = orbit_params.inclination
        
        n = OrbitalMechanics.mean_motion(a)
        p = a * (1 - e**2)
        
        # Rate of change of longitude of ascending node (rad/s)
        omega_dot = -1.5 * n * OrbitalMechanics.J2 * (OrbitalMechanics.EARTH_RADIUS / p)**2 * np.cos(i)
        
        # Rate of change of argument of perigee (rad/s)
        w_dot = 0.75 * n * OrbitalMechanics.J2 * (OrbitalMechanics.EARTH_RADIUS / p)**2 * (5 * np.cos(i)**2 - 1)
        
        return {
            'omega_dot': omega_dot,
            'w_dot': w_dot,
            'period_drift': 2 * np.pi / (n + omega_dot)
        }
    
    @staticmethod
    def relativistic_correction(altitude: float, velocity: float) -> float:
        """Calculate relativistic time dilation effect on satellite clocks"""
        c = 299792458  # speed of light m/s
        
        # Gravitational redshift
        delta_f_grav = OrbitalMechanics.GM_EARTH / (c**2 * (OrbitalMechanics.EARTH_RADIUS + altitude))
        
        # Special relativistic effect
        delta_f_sr = -0.5 * (velocity / c)**2
        
        return delta_f_grav + delta_f_sr
    
class PhysicsInformedLayer(nn.Module):
    """Neural network layer that incorporates orbital mechanics constraints"""
    
    def __init__(self, input_dim: int, hidden_dim: int, orbit_type: str):
        super().__init__()
        self.orbit_type = orbit_type
        self.orbit_params = OrbitalMechanics.ORBIT_PARAMS[orbit_type]
        
        # Standard neural network layers
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Physics-informed parameters
        self.physics_weight = nn.Parameter(torch.tensor(0.1))
        self.orbital_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        # Standard forward pass
        h1 = self.activation(self.linear1(x))
        h2 = self.linear2(h1)
        
        # Physics-informed correction
        physics_correction = self._compute_physics_correction(time_features)
        
        # Combine neural network output with physics
        output = h2 + self.physics_weight * physics_correction + self.orbital_bias
        
        return self.dropout(output)
    
    def _compute_physics_correction(self, time_features: torch.Tensor) -> torch.Tensor:
        """Compute physics-based correction terms"""
        batch_size = time_features.shape[0]
        correction = torch.zeros(batch_size, self.linear2.out_features, device=time_features.device)
        
        # Extract time-related features
        time_of_day = time_features[:, 0]  # Normalized time of day
        orbital_phase = time_features[:, 1]  # Orbital phase
        
        # Orbital period effect
        period = self.orbit_params['period']
        orbital_correction = torch.sin(2 * np.pi * orbital_phase / period)
        
        # Relativistic correction (simplified)
        altitude = self.orbit_params['altitude']
        velocity = 2 * np.pi * (OrbitalMechanics.EARTH_RADIUS + altitude) / period
        rel_correction = OrbitalMechanics.relativistic_correction(altitude, velocity)
        
        # Apply corrections to different dimensions
        correction[:, 0] = orbital_correction
        correction[:, 1] = rel_correction
        
        return correction

class PhysicsInformedGNSSModel(nn.Module):
    """Complete physics-informed model for GNSS error prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 orbit_type: str, num_layers: int = 3):
        super().__init__()
        
        self.orbit_type = orbit_type
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Stack of physics-informed layers
        self.physics_layers = nn.ModuleList([
            PhysicsInformedLayer(hidden_dim, hidden_dim, orbit_type)
            for _ in range(num_layers)
        ])
        
        # Output layers for different horizons
        self.horizon_heads = nn.ModuleDict({
            '15min': nn.Linear(hidden_dim, output_dim),
            '30min': nn.Linear(hidden_dim, output_dim),
            '1hour': nn.Linear(hidden_dim, output_dim),
            '2hour': nn.Linear(hidden_dim, output_dim),
            '24hour': nn.Linear(hidden_dim, output_dim)
        })
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, time_features: torch.Tensor, 
                horizon: str = '1hour') -> Tuple[torch.Tensor, torch.Tensor]:
        # Project input
        h = self.input_projection(x)
        
        # Pass through physics-informed layers
        for layer in self.physics_layers:
            h = layer(h, time_features)
        
        # Generate predictions for specific horizon
        prediction = self.horizon_heads[horizon](h)
        
        # Generate uncertainty estimate
        uncertainty = torch.exp(self.uncertainty_head(h))  # Ensure positive
        
        return prediction, uncertainty
    
    def predict_all_horizons(self, x: torch.Tensor, time_features: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict for all time horizons simultaneously"""
        h = self.input_projection(x)
        
        for layer in self.physics_layers:
            h = layer(h, time_features)
        
        results = {}
        uncertainty = torch.exp(self.uncertainty_head(h))
        
        for horizon in self.horizon_heads.keys():
            prediction = self.horizon_heads[horizon](h)
            results[horizon] = (prediction, uncertainty)
        
        return results

class OrbitalFeatureEngineer:
    """Feature engineering based on orbital mechanics"""
    
    @staticmethod
    def extract_orbital_features(timestamps, sat_ids, orbit_classes):
        """Extract physics-based features from satellite data"""
        features = []
        
        for i, (ts, sat_id, orbit_class) in enumerate(zip(timestamps, sat_ids, orbit_classes)):
            orbit_params = OrbitalMechanics.ORBIT_PARAMS[orbit_class]
            
            # Time-based features
            time_of_day = (ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400
            day_of_year = ts.timetuple().tm_yday / 365.25
            
            # Orbital phase (simplified)
            orbital_period = orbit_params['period']
            orbital_phase = (ts.timestamp() % orbital_period) / orbital_period
            
            # Satellite-specific features
            altitude = orbit_params['altitude']
            inclination = orbit_params['inclination']
            
            # Physics-derived features
            mean_motion = OrbitalMechanics.mean_motion(OrbitalMechanics.EARTH_RADIUS + altitude)
            velocity = 2 * np.pi * (OrbitalMechanics.EARTH_RADIUS + altitude) / orbital_period
            rel_correction = OrbitalMechanics.relativistic_correction(altitude, velocity)
            
            feature_vector = [
                time_of_day,
                day_of_year,
                orbital_phase,
                altitude / 1e6,  # Normalize to millions of meters
                inclination,
                mean_motion * 1e6,  # Scale for numerical stability
                velocity / 1000,  # Convert to km/s
                rel_correction * 1e9,  # Convert to nanoseconds
                hash(sat_id) % 1000 / 1000,  # Satellite ID hash (normalized)
            ]
            
            features.append(feature_vector)
        
        return np.array(features)

def create_physics_informed_model(orbit_type: str, input_dim: int = 64, 
                                hidden_dim: int = 256, output_dim: int = 1) -> PhysicsInformedGNSSModel:
    """Factory function to create physics-informed models for different orbit types"""
    return PhysicsInformedGNSSModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        orbit_type=orbit_type,
        num_layers=4
    )
