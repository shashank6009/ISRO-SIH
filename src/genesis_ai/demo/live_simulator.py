"""
Live GNSS Data Simulator for Demo Mode
Generates realistic satellite data streams for impressive live demonstrations
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
import threading
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class LiveGNSSSimulator:
    """Simulates real-time GNSS satellite data for demo purposes."""
    
    def __init__(self):
        self.satellites = [
            {"id": "IRNSS-1A", "orbit": "GEO/GSO", "lat": 55.0, "lon": 83.0},
            {"id": "IRNSS-1B", "orbit": "GEO/GSO", "lat": 55.0, "lon": 55.0}, 
            {"id": "IRNSS-1C", "orbit": "GEO/GSO", "lat": 55.0, "lon": 111.75},
            {"id": "GSAT-30", "orbit": "MEO", "lat": 20.0, "lon": 78.0},
            {"id": "GSAT-31", "orbit": "MEO", "lat": -20.0, "lon": 78.0},
            {"id": "CARTOSAT-2", "orbit": "LEO", "lat": 0.0, "lon": 78.0},
            {"id": "RESOURCESAT-2", "orbit": "LEO", "lat": 45.0, "lon": 78.0}
        ]
        
        self.is_running = False
        self.data_queue = []
        self.max_queue_size = 1000
        self.update_interval = 15  # seconds
        
        # Simulation parameters
        self.base_errors = {
            "clock": {"mean": 0.05, "std": 0.02, "trend": 0.001},
            "ephem": {"mean": 1.5, "std": 0.5, "trend": 0.01}
        }
        
        # Space weather simulation
        self.space_weather = {
            "kp_index": 2.0,
            "solar_flux": 120.0,
            "geomagnetic_storm": False
        }
        
    def start_simulation(self):
        """Start the live data simulation in a background thread."""
        if self.is_running:
            return
            
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        logger.info("Live GNSS simulation started")
        
    def stop_simulation(self):
        """Stop the live data simulation."""
        self.is_running = False
        logger.info("Live GNSS simulation stopped")
        
    def _simulation_loop(self):
        """Main simulation loop generating realistic satellite data."""
        start_time = datetime.now()
        
        while self.is_running:
            current_time = datetime.now()
            
            # Generate data for each satellite
            for sat in self.satellites:
                # Generate clock and ephemeris errors
                clock_data = self._generate_satellite_data(sat, "clock", current_time)
                ephem_data = self._generate_satellite_data(sat, "ephem", current_time)
                
                self.data_queue.extend([clock_data, ephem_data])
                
            # Simulate space weather effects
            self._update_space_weather(current_time)
            
            # Trim queue if too large
            if len(self.data_queue) > self.max_queue_size:
                self.data_queue = self.data_queue[-self.max_queue_size:]
                
            time.sleep(self.update_interval)
            
    def _generate_satellite_data(self, satellite: Dict, quantity: str, timestamp: datetime) -> Dict:
        """Generate realistic satellite error data with trends and anomalies."""
        base_params = self.base_errors[quantity]
        
        # Base error with random walk
        base_error = np.random.normal(base_params["mean"], base_params["std"])
        
        # Add orbital effects (sinusoidal variation)
        orbital_period = 12 * 3600 if satellite["orbit"] == "GEO/GSO" else 2 * 3600  # seconds
        orbital_phase = (timestamp.timestamp() % orbital_period) / orbital_period * 2 * np.pi
        orbital_effect = 0.1 * base_params["mean"] * np.sin(orbital_phase)
        
        # Add space weather effects
        space_weather_effect = self._calculate_space_weather_effect(satellite, quantity)
        
        # Simulate occasional anomalies (5% chance)
        anomaly_factor = 1.0
        if np.random.random() < 0.05:
            anomaly_factor = np.random.uniform(2.0, 5.0)
            
        final_error = (base_error + orbital_effect + space_weather_effect) * anomaly_factor
        
        return {
            "sat_id": satellite["id"],
            "orbit_class": satellite["orbit"],
            "quantity": quantity,
            "timestamp": timestamp.isoformat(),
            "error": round(final_error, 4),
            "lat": satellite["lat"],
            "lon": satellite["lon"],
            "anomaly": anomaly_factor > 1.5,
            "space_weather_impact": space_weather_effect
        }
        
    def _calculate_space_weather_effect(self, satellite: Dict, quantity: str) -> float:
        """Calculate space weather impact on satellite errors."""
        kp_effect = (self.space_weather["kp_index"] - 2.0) * 0.01
        
        # Higher orbits more affected by space weather
        altitude_factor = {"LEO": 0.5, "MEO": 1.0, "GEO/GSO": 1.5}[satellite["orbit"]]
        
        # Clock errors more sensitive to space weather than ephemeris
        quantity_factor = 2.0 if quantity == "clock" else 1.0
        
        return kp_effect * altitude_factor * quantity_factor
        
    def _update_space_weather(self, current_time: datetime):
        """Simulate realistic space weather variations."""
        # Slowly varying Kp index
        self.space_weather["kp_index"] += np.random.normal(0, 0.1)
        self.space_weather["kp_index"] = np.clip(self.space_weather["kp_index"], 0, 9)
        
        # Solar flux variation
        self.space_weather["solar_flux"] += np.random.normal(0, 2)
        self.space_weather["solar_flux"] = np.clip(self.space_weather["solar_flux"], 80, 300)
        
        # Geomagnetic storm simulation (rare events)
        if np.random.random() < 0.001:  # 0.1% chance per update
            self.space_weather["geomagnetic_storm"] = True
            self.space_weather["kp_index"] = np.random.uniform(6, 9)
        elif self.space_weather["geomagnetic_storm"] and np.random.random() < 0.1:
            self.space_weather["geomagnetic_storm"] = False
            
    def get_latest_data(self, n_points: int = 50) -> pd.DataFrame:
        """Get the latest n data points as a DataFrame."""
        if not self.data_queue:
            return pd.DataFrame()
            
        latest_data = self.data_queue[-n_points:]
        df = pd.DataFrame(latest_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
        
    def get_space_weather_status(self) -> Dict[str, Any]:
        """Get current space weather conditions."""
        severity = "quiet"
        if self.space_weather["kp_index"] > 5:
            severity = "storm"
        elif self.space_weather["kp_index"] > 3:
            severity = "active"
            
        return {
            "kp_index": round(self.space_weather["kp_index"], 1),
            "solar_flux": round(self.space_weather["solar_flux"], 1),
            "geomagnetic_storm": self.space_weather["geomagnetic_storm"],
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        
    def trigger_demo_scenario(self, scenario: str):
        """Trigger specific demo scenarios for presentations."""
        if scenario == "solar_storm":
            self.space_weather["geomagnetic_storm"] = True
            self.space_weather["kp_index"] = 7.5
            logger.info("Demo: Solar storm scenario activated")
            
        elif scenario == "satellite_anomaly":
            # Inject anomaly into random satellite
            if self.satellites:
                sat = np.random.choice(self.satellites)
                # This will be picked up in next data generation
                logger.info(f"Demo: Satellite anomaly scenario for {sat['id']}")
                
        elif scenario == "normal_operations":
            self.space_weather["geomagnetic_storm"] = False
            self.space_weather["kp_index"] = 2.0
            logger.info("Demo: Normal operations scenario")


class DemoController:
    """Controls demo scenarios and presentation modes."""
    
    def __init__(self):
        self.simulator = LiveGNSSSimulator()
        self.current_scenario = "normal_operations"
        self.demo_scripts = {
            "mission_control": self._mission_control_demo,
            "anomaly_detection": self._anomaly_detection_demo,
            "space_weather": self._space_weather_demo,
            "prediction_accuracy": self._prediction_accuracy_demo
        }
        
    def start_demo(self, scenario: str = "mission_control"):
        """Start a specific demo scenario."""
        self.simulator.start_simulation()
        if scenario in self.demo_scripts:
            threading.Thread(target=self.demo_scripts[scenario], daemon=True).start()
            
    def _mission_control_demo(self):
        """Demo script: Mission control operations."""
        time.sleep(5)
        self.simulator.trigger_demo_scenario("normal_operations")
        
    def _anomaly_detection_demo(self):
        """Demo script: Anomaly detection capabilities."""
        time.sleep(10)
        self.simulator.trigger_demo_scenario("satellite_anomaly")
        time.sleep(30)
        self.simulator.trigger_demo_scenario("normal_operations")
        
    def _space_weather_demo(self):
        """Demo script: Space weather impact demonstration."""
        time.sleep(15)
        self.simulator.trigger_demo_scenario("solar_storm")
        time.sleep(45)
        self.simulator.trigger_demo_scenario("normal_operations")
        
    def _prediction_accuracy_demo(self):
        """Demo script: Prediction accuracy showcase."""
        # Generate historical data for comparison
        pass


# Global simulator instance for use across the application
global_simulator = LiveGNSSSimulator()
demo_controller = DemoController()

def get_live_data(n_points: int = 50) -> pd.DataFrame:
    """Get live simulated data for the application."""
    return global_simulator.get_latest_data(n_points)

def get_space_weather() -> Dict[str, Any]:
    """Get current space weather status."""
    return global_simulator.get_space_weather_status()

def start_live_demo():
    """Start the live demo mode."""
    global_simulator.start_simulation()
    
def stop_live_demo():
    """Stop the live demo mode."""
    global_simulator.stop_simulation()
