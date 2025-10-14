"""
Competition-Specific Database Models for GNSS Error Prediction
Handles broadcast vs modeled value differences for clock bias and ephemeris
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel
import pandas as pd

Base = declarative_base()

class CompetitionGNSSRecord(Base):
    """Database model for competition GNSS data with broadcast vs modeled values"""
    __tablename__ = 'competition_gnss_records'
    
    id = Column(Integer, primary_key=True)
    satellite_id = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    orbit_class = Column(String(10), nullable=False, index=True)  # GEO/GSO or MEO
    
    # Clock Bias Data (broadcast vs modeled)
    broadcast_clock_bias = Column(Float, nullable=False)
    modeled_clock_bias = Column(Float, nullable=False) 
    clock_error = Column(Float, nullable=False)  # broadcast - modeled
    
    # Ephemeris Data (broadcast vs modeled)
    broadcast_ephemeris = Column(JSON, nullable=False)  # {x, y, z, vx, vy, vz}
    modeled_ephemeris = Column(JSON, nullable=False)    # {x, y, z, vx, vy, vz}
    ephemeris_errors = Column(JSON, nullable=False)     # {dx, dy, dz, dvx, dvy, dvz}
    
    # Additional metadata
    data_quality = Column(Float, default=1.0)
    is_training = Column(Integer, default=1)  # 1 for 7-day training, 0 for day-8 test
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_sat_time', 'satellite_id', 'timestamp'),
        Index('idx_orbit_time', 'orbit_class', 'timestamp'),
        Index('idx_training_flag', 'is_training', 'timestamp'),
    )

class CompetitionPrediction(Base):
    """Database model for storing competition predictions"""
    __tablename__ = 'competition_predictions'
    
    id = Column(Integer, primary_key=True)
    satellite_id = Column(String(50), nullable=False, index=True)
    prediction_timestamp = Column(DateTime, nullable=False, index=True)
    target_timestamp = Column(DateTime, nullable=False, index=True)
    orbit_class = Column(String(10), nullable=False)
    
    # Prediction horizon
    horizon_minutes = Column(Integer, nullable=False, index=True)  # 15, 30, 60, 120, 1440
    
    # Clock bias predictions
    predicted_clock_error = Column(Float, nullable=False)
    clock_error_uncertainty = Column(Float, nullable=False)
    
    # Ephemeris predictions
    predicted_ephemeris_errors = Column(JSON, nullable=False)  # {dx, dy, dz, dvx, dvy, dvz}
    ephemeris_uncertainties = Column(JSON, nullable=False)     # {dx_std, dy_std, ...}
    
    # Model information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Quality metrics
    prediction_confidence = Column(Float, nullable=False)
    normality_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic models for API
class CompetitionDataPoint(BaseModel):
    """Pydantic model for competition data input"""
    satellite_id: str
    timestamp: datetime
    orbit_class: str  # "GEO", "GSO", or "MEO"
    
    # Clock data
    broadcast_clock_bias: float
    modeled_clock_bias: float
    clock_error: Optional[float] = None  # Will be computed if not provided
    
    # Ephemeris data
    broadcast_ephemeris: Dict[str, float]  # {x, y, z, vx, vy, vz}
    modeled_ephemeris: Dict[str, float]    # {x, y, z, vx, vy, vz}
    ephemeris_errors: Optional[Dict[str, float]] = None  # Will be computed if not provided
    
    data_quality: float = 1.0
    is_training: bool = True
    
    def compute_errors(self):
        """Compute errors if not provided"""
        if self.clock_error is None:
            self.clock_error = self.broadcast_clock_bias - self.modeled_clock_bias
            
        if self.ephemeris_errors is None:
            self.ephemeris_errors = {
                'dx': self.broadcast_ephemeris['x'] - self.modeled_ephemeris['x'],
                'dy': self.broadcast_ephemeris['y'] - self.modeled_ephemeris['y'],
                'dz': self.broadcast_ephemeris['z'] - self.modeled_ephemeris['z'],
                'dvx': self.broadcast_ephemeris['vx'] - self.modeled_ephemeris['vx'],
                'dvy': self.broadcast_ephemeris['vy'] - self.modeled_ephemeris['vy'],
                'dvz': self.broadcast_ephemeris['vz'] - self.modeled_ephemeris['vz'],
            }

class CompetitionPredictionRequest(BaseModel):
    """Request model for competition predictions"""
    training_data: list[CompetitionDataPoint]
    prediction_horizons: list[int] = [15, 30, 60, 120, 1440]  # minutes
    target_satellites: Optional[list[str]] = None
    orbit_classes: Optional[list[str]] = None

class CompetitionPredictionResponse(BaseModel):
    """Response model for competition predictions"""
    satellite_id: str
    orbit_class: str
    predictions: Dict[str, Dict[str, Any]]  # {horizon: {clock: {...}, ephemeris: {...}}}
    normality_scores: Dict[str, float]      # {horizon: normality_score}
    overall_normality: float
    prediction_timestamp: datetime
    model_confidence: float

class CompetitionDataLoader:
    """Specialized data loader for competition format"""
    
    @staticmethod
    def load_competition_csv(filepath: str) -> pd.DataFrame:
        """Load competition data from CSV with expected format"""
        
        # Expected columns for competition data
        required_columns = [
            'satellite_id', 'timestamp', 'orbit_class',
            'broadcast_clock_bias', 'modeled_clock_bias', 'clock_error',
            'broadcast_x', 'broadcast_y', 'broadcast_z',
            'broadcast_vx', 'broadcast_vy', 'broadcast_vz',
            'modeled_x', 'modeled_y', 'modeled_z',
            'modeled_vx', 'modeled_vy', 'modeled_vz',
            'ephemeris_error_x', 'ephemeris_error_y', 'ephemeris_error_z',
            'ephemeris_error_vx', 'ephemeris_error_vy', 'ephemeris_error_vz'
        ]
        
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        
        # Validate required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate orbit classes
        valid_orbits = {'GEO', 'GSO', 'MEO'}
        invalid_orbits = set(df['orbit_class'].unique()) - valid_orbits
        if invalid_orbits:
            raise ValueError(f"Invalid orbit classes found: {invalid_orbits}. Expected: {valid_orbits}")
        
        return df
    
    @staticmethod
    def prepare_competition_data(df: pd.DataFrame) -> list[CompetitionDataPoint]:
        """Convert DataFrame to competition data points"""
        
        data_points = []
        
        for _, row in df.iterrows():
            # Construct ephemeris dictionaries
            broadcast_eph = {
                'x': row['broadcast_x'], 'y': row['broadcast_y'], 'z': row['broadcast_z'],
                'vx': row['broadcast_vx'], 'vy': row['broadcast_vy'], 'vz': row['broadcast_vz']
            }
            
            modeled_eph = {
                'x': row['modeled_x'], 'y': row['modeled_y'], 'z': row['modeled_z'],
                'vx': row['modeled_vx'], 'vy': row['modeled_vy'], 'vz': row['modeled_vz']
            }
            
            ephemeris_errors = {
                'dx': row['ephemeris_error_x'], 'dy': row['ephemeris_error_y'], 'dz': row['ephemeris_error_z'],
                'dvx': row['ephemeris_error_vx'], 'dvy': row['ephemeris_error_vy'], 'dvz': row['ephemeris_error_vz']
            }
            
            data_point = CompetitionDataPoint(
                satellite_id=row['satellite_id'],
                timestamp=row['timestamp'],
                orbit_class=row['orbit_class'],
                broadcast_clock_bias=row['broadcast_clock_bias'],
                modeled_clock_bias=row['modeled_clock_bias'],
                clock_error=row['clock_error'],
                broadcast_ephemeris=broadcast_eph,
                modeled_ephemeris=modeled_eph,
                ephemeris_errors=ephemeris_errors,
                data_quality=row.get('data_quality', 1.0),
                is_training=row.get('is_training', True)
            )
            
            data_points.append(data_point)
        
        return data_points

# Competition-specific utility functions
def separate_by_orbit_class(data_points: list[CompetitionDataPoint]) -> Dict[str, list[CompetitionDataPoint]]:
    """Separate data points by orbit class for specialized processing"""
    separated = {'GEO': [], 'GSO': [], 'MEO': []}
    
    for point in data_points:
        if point.orbit_class in separated:
            separated[point.orbit_class].append(point)
        else:
            # Handle GSO as GEO for modeling purposes
            if point.orbit_class == 'GSO':
                separated['GEO'].append(point)
    
    return separated

def extract_day8_targets(data_points: list[CompetitionDataPoint]) -> list[CompetitionDataPoint]:
    """Extract day-8 data points for prediction targets"""
    return [point for point in data_points if not point.is_training]

def extract_training_data(data_points: list[CompetitionDataPoint]) -> list[CompetitionDataPoint]:
    """Extract 7-day training data"""
    return [point for point in data_points if point.is_training]
