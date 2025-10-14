import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SpaceWeatherFeed:
    """Real-time space weather data feed for GNSS error correlation."""
    
    def __init__(self):
        self.base_urls = {
            "noaa_kp": "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json",
            "noaa_dst": "https://services.swpc.noaa.gov/json/geospace/geomag_dst_1m.json",
            "noaa_f107": "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
        }
    
    def fetch_space_weather(self) -> Dict[str, Any]:
        """
        Pull latest Kp, Dst, and F10.7 indices from NOAA SWPC.
        These directly correlate with GNSS ionospheric delays.
        """
        try:
            # Fetch Kp index (geomagnetic activity)
            kp_data = requests.get(self.base_urls["noaa_kp"], timeout=10).json()
            kp_df = pd.DataFrame(kp_data)
            kp_df["time_tag"] = pd.to_datetime(kp_df["time_tag"])
            latest_kp = kp_df.sort_values("time_tag").tail(1).iloc[0]
            
            # Fetch Dst index (ring current strength)
            try:
                dst_data = requests.get(self.base_urls["noaa_dst"], timeout=10).json()
                dst_df = pd.DataFrame(dst_data)
                dst_df["time_tag"] = pd.to_datetime(dst_df["time_tag"])
                latest_dst = dst_df.sort_values("time_tag").tail(1).iloc[0]
                dst_value = float(latest_dst.get("dst", 0))
            except:
                dst_value = None
            
            # Calculate space weather severity
            kp_val = float(latest_kp["kp_index"])
            severity = self._calculate_severity(kp_val, dst_value)
            
            return {
                "kp_index": kp_val,
                "dst_index": dst_value,
                "severity": severity,
                "timestamp": latest_kp["time_tag"].isoformat(),
                "status": "active",
                "impact_level": self._assess_gnss_impact(kp_val, dst_value)
            }
            
        except Exception as e:
            logger.error(f"Space weather fetch failed: {e}")
            return {
                "error": str(e),
                "status": "offline",
                "fallback_data": self._get_fallback_space_weather()
            }
    
    def _calculate_severity(self, kp: float, dst: Optional[float]) -> str:
        """Calculate space weather severity level."""
        if kp >= 7:
            return "severe"
        elif kp >= 5:
            return "moderate" 
        elif kp >= 3:
            return "minor"
        else:
            return "quiet"
    
    def _assess_gnss_impact(self, kp: float, dst: Optional[float]) -> str:
        """Assess potential GNSS impact from space weather."""
        if kp >= 6:
            return "high"
        elif kp >= 4:
            return "medium"
        elif kp >= 2:
            return "low"
        else:
            return "minimal"
    
    def _get_fallback_space_weather(self) -> Dict[str, Any]:
        """Provide fallback data when live feeds are unavailable."""
        return {
            "kp_index": 2.0,  # Typical quiet conditions
            "dst_index": -15.0,
            "severity": "quiet",
            "timestamp": datetime.utcnow().isoformat(),
            "note": "Fallback data - live feed unavailable"
        }

class IonosphereFeed:
    """Ionospheric condition monitoring for GNSS error prediction."""
    
    def __init__(self):
        # In production, this would connect to ISRO's ionospheric monitoring network
        self.mock_stations = [
            {"name": "Bengaluru", "lat": 12.97, "lon": 77.59},
            {"name": "Lucknow", "lat": 26.85, "lon": 80.95},
            {"name": "Shillong", "lat": 25.57, "lon": 91.88},
            {"name": "Thiruvananthapuram", "lat": 8.52, "lon": 76.94}
        ]
    
    def fetch_ionosphere_data(self) -> Dict[str, Any]:
        """
        Fetch ionospheric Total Electron Content (TEC) data.
        In production, this would connect to ISRO's ionospheric monitoring network.
        """
        try:
            # Simulate realistic TEC values based on time of day and location
            current_hour = datetime.utcnow().hour
            
            # TEC varies with solar activity and local time
            base_tec = 15.0 + 10.0 * np.sin((current_hour - 6) * np.pi / 12)  # Peak at noon
            base_tec = max(5.0, base_tec)  # Minimum nighttime TEC
            
            # Add some realistic variation
            tec_variation = np.random.normal(0, 2.0)
            avg_tec = base_tec + tec_variation
            
            # Calculate TEC gradient (important for differential GNSS)
            tec_gradient = np.random.normal(0.5, 0.2)
            
            return {
                "TEC_avg": round(avg_tec, 2),
                "TEC_gradient": round(tec_gradient, 3),
                "stations_active": len(self.mock_stations),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "operational",
                "quality": "good" if abs(tec_gradient) < 1.0 else "degraded",
                "source": "ISRO Ionospheric Network (simulated)"
            }
            
        except Exception as e:
            logger.error(f"Ionosphere data fetch failed: {e}")
            return {
                "error": str(e),
                "status": "offline",
                "fallback_data": {
                    "TEC_avg": 12.5,
                    "TEC_gradient": 0.3,
                    "note": "Fallback data"
                }
            }
    
    def get_station_data(self) -> pd.DataFrame:
        """Get ISRO ground station locations for mapping."""
        stations_data = []
        
        for station in self.mock_stations:
            # Simulate station-specific TEC measurements
            local_tec = 12.0 + np.random.normal(0, 3.0)
            status = "online" if np.random.random() > 0.1 else "maintenance"
            
            stations_data.append({
                "station": station["name"],
                "lat": station["lat"],
                "lon": station["lon"],
                "TEC": round(local_tec, 2),
                "status": status,
                "type": "ionospheric_monitor"
            })
        
        return pd.DataFrame(stations_data)

class SolarActivityFeed:
    """Solar activity monitoring for long-term GNSS trend analysis."""
    
    def fetch_solar_flux(self) -> Dict[str, Any]:
        """
        Fetch F10.7 solar flux data.
        High solar activity increases ionospheric variability.
        """
        try:
            # Simulate F10.7 flux (typical range: 70-300 SFU)
            # Higher values indicate more solar activity
            base_flux = 120.0 + 30.0 * np.sin(datetime.utcnow().timetuple().tm_yday * 2 * np.pi / 365)
            flux_noise = np.random.normal(0, 10.0)
            f107_flux = max(70.0, base_flux + flux_noise)
            
            # Assess solar activity level
            if f107_flux > 200:
                activity_level = "very_high"
            elif f107_flux > 150:
                activity_level = "high"
            elif f107_flux > 100:
                activity_level = "moderate"
            else:
                activity_level = "low"
            
            return {
                "f107_flux": round(f107_flux, 1),
                "activity_level": activity_level,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "active",
                "trend": "stable"  # Could be "increasing", "decreasing", "stable"
            }
            
        except Exception as e:
            logger.error(f"Solar flux fetch failed: {e}")
            return {
                "error": str(e),
                "status": "offline",
                "fallback_data": {
                    "f107_flux": 120.0,
                    "activity_level": "moderate"
                }
            }

# Convenience functions for backward compatibility
def fetch_space_weather() -> Dict[str, Any]:
    """Convenience function to fetch space weather data."""
    feed = SpaceWeatherFeed()
    return feed.fetch_space_weather()

def fetch_ionosphere_data() -> Dict[str, Any]:
    """Convenience function to fetch ionosphere data."""
    feed = IonosphereFeed()
    return feed.fetch_ionosphere_data()

def get_ground_stations() -> pd.DataFrame:
    """Get ISRO ground station data for visualization."""
    feed = IonosphereFeed()
    return feed.get_station_data()

def fetch_solar_activity() -> Dict[str, Any]:
    """Convenience function to fetch solar activity data."""
    feed = SolarActivityFeed()
    return feed.fetch_solar_flux()

# Environmental data aggregator
class EnvironmentalMonitor:
    """Aggregate all environmental feeds for comprehensive monitoring."""
    
    def __init__(self):
        self.space_weather = SpaceWeatherFeed()
        self.ionosphere = IonosphereFeed()
        self.solar = SolarActivityFeed()
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive environmental status for GNSS operations."""
        sw_data = self.space_weather.fetch_space_weather()
        ion_data = self.ionosphere.fetch_ionosphere_data()
        solar_data = self.solar.fetch_solar_flux()
        
        # Calculate overall GNSS environment health
        health_score = self._calculate_environment_health(sw_data, ion_data, solar_data)
        
        return {
            "space_weather": sw_data,
            "ionosphere": ion_data,
            "solar_activity": solar_data,
            "overall_health": health_score,
            "timestamp": datetime.utcnow().isoformat(),
            "recommendations": self._get_operational_recommendations(health_score)
        }
    
    def _calculate_environment_health(self, sw: Dict, ion: Dict, solar: Dict) -> Dict[str, Any]:
        """Calculate overall environmental health score for GNSS operations."""
        score = 100.0
        
        # Penalize high geomagnetic activity
        if "kp_index" in sw:
            kp = sw["kp_index"]
            if kp > 5:
                score -= (kp - 5) * 15
            elif kp > 3:
                score -= (kp - 3) * 5
        
        # Penalize high TEC gradients
        if "TEC_gradient" in ion and abs(ion["TEC_gradient"]) > 1.0:
            score -= abs(ion["TEC_gradient"]) * 10
        
        # Penalize very high solar activity
        if "f107_flux" in solar and solar["f107_flux"] > 200:
            score -= (solar["f107_flux"] - 200) * 0.2
        
        score = max(0, min(100, score))
        
        if score >= 80:
            status = "excellent"
        elif score >= 60:
            status = "good"
        elif score >= 40:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "score": round(score, 1),
            "status": status,
            "factors": {
                "geomagnetic": sw.get("severity", "unknown"),
                "ionospheric": ion.get("quality", "unknown"),
                "solar": solar.get("activity_level", "unknown")
            }
        }
    
    def _get_operational_recommendations(self, health: Dict[str, Any]) -> list:
        """Provide operational recommendations based on environmental conditions."""
        recommendations = []
        
        if health["score"] < 60:
            recommendations.append("Consider increased GNSS monitoring frequency")
            
        if health["factors"]["geomagnetic"] in ["severe", "moderate"]:
            recommendations.append("Expect increased ionospheric delays")
            
        if health["factors"]["ionospheric"] == "degraded":
            recommendations.append("Monitor differential GNSS performance closely")
            
        if health["factors"]["solar"] == "very_high":
            recommendations.append("Prepare for potential communication disruptions")
        
        if not recommendations:
            recommendations.append("Nominal GNSS operating conditions")
        
        return recommendations

if __name__ == "__main__":
    # Test the environmental feeds
    print("🌦️ Testing GENESIS-AI Environmental Feeds...")
    
    monitor = EnvironmentalMonitor()
    status = monitor.get_comprehensive_status()
    
    print(f"Space Weather: Kp={status['space_weather'].get('kp_index', 'N/A')}")
    print(f"Ionosphere: TEC={status['ionosphere'].get('TEC_avg', 'N/A')} TECU")
    print(f"Solar Activity: F10.7={status['solar_activity'].get('f107_flux', 'N/A')} SFU")
    print(f"Overall Health: {status['overall_health']['score']}/100 ({status['overall_health']['status']})")
    print("Recommendations:")
    for rec in status["recommendations"]:
        print(f"  • {rec}")
    
    print("\n✅ Environmental feed test completed!")

