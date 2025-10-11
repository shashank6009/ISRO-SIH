"""
Automated Anomaly Detection System for GENESIS-AI
Advanced anomaly detection with predictive alerts and trend analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    TREND_DEVIATION = "trend_deviation"
    PATTERN_BREAK = "pattern_break"
    CORRELATION_ANOMALY = "correlation_anomaly"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SPACE_WEATHER_IMPACT = "space_weather_impact"

class SeverityLevel(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Anomaly:
    """Anomaly detection result."""
    satellite_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    confidence: float
    description: str
    detected_at: datetime
    value: float
    expected_range: Tuple[float, float]
    recommendation: str
    metadata: Dict[str, Any]

class SatelliteAnomalyDetector:
    """Advanced anomaly detection for satellite GNSS errors."""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Historical data for trend analysis
        self.historical_data = {}
        self.baseline_stats = {}
        
        # Anomaly thresholds
        self.thresholds = {
            "error_magnitude": {"low": 0.05, "medium": 0.1, "high": 0.2, "critical": 0.5},
            "error_rate_change": {"low": 0.1, "medium": 0.25, "high": 0.5, "critical": 1.0},
            "prediction_accuracy": {"low": 0.9, "medium": 0.8, "high": 0.7, "critical": 0.6}
        }
        
    def fit(self, training_data: pd.DataFrame):
        """Train the anomaly detection models on historical data."""
        logger.info("Training anomaly detection models...")
        
        # Prepare features for training
        features = self._extract_features(training_data)
        
        if len(features) == 0:
            logger.warning("No features extracted for training")
            return
            
        # Fit isolation forest
        X = self.scaler.fit_transform(features)
        self.isolation_forest.fit(X)
        
        # Calculate baseline statistics for each satellite
        self._calculate_baseline_stats(training_data)
        
        self.is_fitted = True
        logger.info(f"Anomaly detection models trained on {len(training_data)} samples")
        
    def detect_anomalies(self, current_data: pd.DataFrame, 
                        predictions: Optional[pd.DataFrame] = None) -> List[Anomaly]:
        """
        Detect anomalies in current satellite data.
        
        Args:
            current_data: Current satellite error measurements
            predictions: Model predictions for comparison
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        for satellite_id in current_data['sat_id'].unique():
            sat_data = current_data[current_data['sat_id'] == satellite_id]
            sat_predictions = predictions[predictions['sat_id'] == satellite_id] if predictions is not None else None
            
            # Run different anomaly detection methods
            anomalies.extend(self._detect_statistical_outliers(satellite_id, sat_data))
            anomalies.extend(self._detect_trend_deviations(satellite_id, sat_data))
            anomalies.extend(self._detect_pattern_breaks(satellite_id, sat_data))
            
            if sat_predictions is not None:
                anomalies.extend(self._detect_prediction_anomalies(satellite_id, sat_data, sat_predictions))
                
        return sorted(anomalies, key=lambda x: (x.severity.value, x.confidence), reverse=True)
        
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for anomaly detection."""
        features = []
        
        for satellite_id in data['sat_id'].unique():
            sat_data = data[data['sat_id'] == satellite_id].sort_values('timestamp')
            
            if len(sat_data) < 5:  # Need minimum data points
                continue
                
            # Statistical features
            error_values = sat_data['error'].values
            
            feature_vector = [
                np.mean(error_values),           # Mean error
                np.std(error_values),            # Error variability
                np.min(error_values),            # Minimum error
                np.max(error_values),            # Maximum error
                np.percentile(error_values, 95), # 95th percentile
                len(error_values),               # Number of measurements
            ]
            
            # Trend features
            if len(error_values) > 1:
                trend = np.polyfit(range(len(error_values)), error_values, 1)[0]
                feature_vector.append(trend)
            else:
                feature_vector.append(0)
                
            # Temporal features
            time_diffs = sat_data['timestamp'].diff().dt.total_seconds().fillna(0)
            feature_vector.extend([
                np.mean(time_diffs),  # Average time between measurements
                np.std(time_diffs)    # Variability in measurement intervals
            ])
            
            features.append(feature_vector)
            
        return np.array(features) if features else np.array([]).reshape(0, 9)
        
    def _calculate_baseline_stats(self, training_data: pd.DataFrame):
        """Calculate baseline statistics for each satellite."""
        self.baseline_stats = {}
        
        for satellite_id in training_data['sat_id'].unique():
            sat_data = training_data[training_data['sat_id'] == satellite_id]
            error_values = sat_data['error'].values
            
            if len(error_values) == 0:
                continue
                
            self.baseline_stats[satellite_id] = {
                "mean_error": np.mean(error_values),
                "std_error": np.std(error_values),
                "min_error": np.min(error_values),
                "max_error": np.max(error_values),
                "percentile_95": np.percentile(error_values, 95),
                "percentile_5": np.percentile(error_values, 5),
                "typical_range": (
                    np.percentile(error_values, 5),
                    np.percentile(error_values, 95)
                )
            }
            
    def _detect_statistical_outliers(self, satellite_id: str, data: pd.DataFrame) -> List[Anomaly]:
        """Detect statistical outliers using isolation forest and z-score."""
        anomalies = []
        
        if not self.is_fitted or satellite_id not in self.baseline_stats:
            return anomalies
            
        baseline = self.baseline_stats[satellite_id]
        
        for _, row in data.iterrows():
            error_value = row['error']
            
            # Z-score based detection
            z_score = abs(error_value - baseline["mean_error"]) / (baseline["std_error"] + 1e-8)
            
            if z_score > 3:  # 3-sigma rule
                severity = self._determine_severity_from_zscore(z_score)
                
                anomalies.append(Anomaly(
                    satellite_id=satellite_id,
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=severity,
                    confidence=min(0.99, z_score / 5),  # Confidence based on z-score
                    description=f"Error value {error_value:.4f} is {z_score:.1f} standard deviations from normal",
                    detected_at=datetime.now(),
                    value=error_value,
                    expected_range=baseline["typical_range"],
                    recommendation=self._get_outlier_recommendation(severity),
                    metadata={"z_score": z_score, "baseline_mean": baseline["mean_error"]}
                ))
                
        return anomalies
        
    def _detect_trend_deviations(self, satellite_id: str, data: pd.DataFrame) -> List[Anomaly]:
        """Detect deviations from expected trends."""
        anomalies = []
        
        if len(data) < 10:  # Need sufficient data for trend analysis
            return anomalies
            
        # Sort by timestamp
        data_sorted = data.sort_values('timestamp')
        error_values = data_sorted['error'].values
        
        # Calculate recent trend
        recent_window = min(10, len(error_values))
        recent_errors = error_values[-recent_window:]
        
        if len(recent_errors) < 3:
            return anomalies
            
        # Linear trend in recent data
        recent_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
        
        # Compare with baseline if available
        if satellite_id in self.baseline_stats:
            baseline = self.baseline_stats[satellite_id]
            expected_trend = 0  # Assume stable errors as baseline
            
            trend_deviation = abs(recent_trend - expected_trend)
            
            # Check if trend is significantly different
            if trend_deviation > 0.01:  # Threshold for significant trend change
                severity = self._determine_severity_from_trend(trend_deviation)
                
                anomalies.append(Anomaly(
                    satellite_id=satellite_id,
                    anomaly_type=AnomalyType.TREND_DEVIATION,
                    severity=severity,
                    confidence=min(0.95, trend_deviation * 50),
                    description=f"Detected {'increasing' if recent_trend > 0 else 'decreasing'} error trend: {recent_trend:.6f} per measurement",
                    detected_at=datetime.now(),
                    value=recent_trend,
                    expected_range=(-0.01, 0.01),
                    recommendation=self._get_trend_recommendation(recent_trend, severity),
                    metadata={"trend_slope": recent_trend, "window_size": recent_window}
                ))
                
        return anomalies
        
    def _detect_pattern_breaks(self, satellite_id: str, data: pd.DataFrame) -> List[Anomaly]:
        """Detect breaks in expected patterns (e.g., orbital cycles)."""
        anomalies = []
        
        if len(data) < 20:  # Need sufficient data for pattern analysis
            return anomalies
            
        # Sort by timestamp
        data_sorted = data.sort_values('timestamp')
        error_values = data_sorted['error'].values
        
        # Look for sudden changes in variance
        window_size = 10
        variances = []
        
        for i in range(len(error_values) - window_size + 1):
            window_data = error_values[i:i + window_size]
            variances.append(np.var(window_data))
            
        if len(variances) < 2:
            return anomalies
            
        # Detect sudden variance changes
        variance_changes = np.diff(variances)
        
        for i, change in enumerate(variance_changes):
            if abs(change) > np.std(variances) * 2:  # Significant variance change
                severity = self._determine_severity_from_variance_change(abs(change))
                
                anomalies.append(Anomaly(
                    satellite_id=satellite_id,
                    anomaly_type=AnomalyType.PATTERN_BREAK,
                    severity=severity,
                    confidence=min(0.9, abs(change) / np.std(variances) / 5),
                    description=f"Detected pattern break: variance changed by {change:.6f}",
                    detected_at=datetime.now(),
                    value=change,
                    expected_range=(-np.std(variances), np.std(variances)),
                    recommendation="Investigate potential hardware or environmental changes",
                    metadata={"variance_change": change, "window_position": i}
                ))
                
        return anomalies
        
    def _detect_prediction_anomalies(self, satellite_id: str, actual_data: pd.DataFrame, 
                                   predictions: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies by comparing actual vs predicted values."""
        anomalies = []
        
        # Merge actual and predicted data
        merged = pd.merge(actual_data, predictions, on=['sat_id', 'timestamp'], 
                         suffixes=('_actual', '_predicted'), how='inner')
        
        for _, row in merged.iterrows():
            actual_error = row['error_actual']
            predicted_error = row['error_predicted']
            
            # Calculate prediction residual
            residual = abs(actual_error - predicted_error)
            
            # Get prediction bounds if available
            lower_bound = row.get('lower_bound', predicted_error - 0.05)
            upper_bound = row.get('upper_bound', predicted_error + 0.05)
            
            # Check if actual value is outside prediction bounds
            if actual_error < lower_bound or actual_error > upper_bound:
                severity = self._determine_severity_from_prediction_error(residual)
                
                anomalies.append(Anomaly(
                    satellite_id=satellite_id,
                    anomaly_type=AnomalyType.CORRELATION_ANOMALY,
                    severity=severity,
                    confidence=min(0.95, residual / 0.1),
                    description=f"Actual error {actual_error:.4f} outside predicted range [{lower_bound:.4f}, {upper_bound:.4f}]",
                    detected_at=datetime.now(),
                    value=actual_error,
                    expected_range=(lower_bound, upper_bound),
                    recommendation="Review model accuracy and potential new error sources",
                    metadata={
                        "predicted_error": predicted_error,
                        "residual": residual,
                        "prediction_bounds": (lower_bound, upper_bound)
                    }
                ))
                
        return anomalies
        
    def _determine_severity_from_zscore(self, z_score: float) -> SeverityLevel:
        """Determine severity based on z-score."""
        if z_score > 5:
            return SeverityLevel.CRITICAL
        elif z_score > 4:
            return SeverityLevel.HIGH
        elif z_score > 3:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
            
    def _determine_severity_from_trend(self, trend_deviation: float) -> SeverityLevel:
        """Determine severity based on trend deviation."""
        if trend_deviation > 0.05:
            return SeverityLevel.CRITICAL
        elif trend_deviation > 0.03:
            return SeverityLevel.HIGH
        elif trend_deviation > 0.02:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
            
    def _determine_severity_from_variance_change(self, variance_change: float) -> SeverityLevel:
        """Determine severity based on variance change."""
        if variance_change > 0.01:
            return SeverityLevel.HIGH
        elif variance_change > 0.005:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
            
    def _determine_severity_from_prediction_error(self, residual: float) -> SeverityLevel:
        """Determine severity based on prediction error."""
        if residual > 0.2:
            return SeverityLevel.CRITICAL
        elif residual > 0.1:
            return SeverityLevel.HIGH
        elif residual > 0.05:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
            
    def _get_outlier_recommendation(self, severity: SeverityLevel) -> str:
        """Get recommendation for statistical outliers."""
        recommendations = {
            SeverityLevel.LOW: "Monitor for recurring patterns",
            SeverityLevel.MEDIUM: "Investigate potential causes and verify measurements",
            SeverityLevel.HIGH: "Immediate investigation required - check satellite health",
            SeverityLevel.CRITICAL: "URGENT: Potential satellite malfunction - initiate emergency protocols"
        }
        return recommendations.get(severity, "Monitor situation")
        
    def _get_trend_recommendation(self, trend: float, severity: SeverityLevel) -> str:
        """Get recommendation for trend deviations."""
        if trend > 0:
            direction = "degrading"
        else:
            direction = "improving"
            
        if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            return f"Satellite performance is {direction} rapidly - investigate hardware status"
        else:
            return f"Monitor {direction} trend - may indicate gradual changes"
            
    def generate_anomaly_report(self, anomalies: List[Anomaly]) -> str:
        """Generate comprehensive anomaly report."""
        if not anomalies:
            return "✅ No anomalies detected - all satellites operating normally"
            
        report = f"🚨 ANOMALY DETECTION REPORT\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        report += f"Total Anomalies: {len(anomalies)}\n\n"
        
        # Group by severity
        by_severity = {}
        for anomaly in anomalies:
            severity = anomaly.severity.value
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(anomaly)
            
        # Report by severity
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in by_severity:
                count = len(by_severity[severity])
                report += f"🔴 {severity.upper()}: {count} anomalies\n"
                
                for anomaly in by_severity[severity][:3]:  # Show top 3 per severity
                    report += f"  • {anomaly.satellite_id}: {anomaly.description}\n"
                    report += f"    Recommendation: {anomaly.recommendation}\n"
                    
                if len(by_severity[severity]) > 3:
                    report += f"    ... and {len(by_severity[severity]) - 3} more\n"
                report += "\n"
                
        return report


class PredictiveAnomalySystem:
    """Predictive anomaly detection system with early warning capabilities."""
    
    def __init__(self):
        self.detector = SatelliteAnomalyDetector()
        self.alert_history = []
        self.prediction_horizon = 24  # hours
        
    def predict_future_anomalies(self, current_data: pd.DataFrame, 
                                predictions: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict potential future anomalies based on current trends."""
        future_anomalies = []
        
        for satellite_id in current_data['sat_id'].unique():
            sat_data = current_data[current_data['sat_id'] == satellite_id]
            sat_predictions = predictions[predictions['sat_id'] == satellite_id]
            
            # Analyze prediction trends
            if len(sat_predictions) > 5:
                pred_errors = sat_predictions['predicted_error'].values
                
                # Check for concerning trends in predictions
                trend = np.polyfit(range(len(pred_errors)), pred_errors, 1)[0]
                
                if abs(trend) > 0.005:  # Significant trend in predictions
                    risk_level = "high" if abs(trend) > 0.01 else "medium"
                    
                    future_anomalies.append({
                        "satellite_id": satellite_id,
                        "risk_type": "performance_degradation",
                        "risk_level": risk_level,
                        "estimated_time": "6-12 hours",
                        "description": f"Predicted {'increasing' if trend > 0 else 'decreasing'} error trend",
                        "preventive_action": "Schedule maintenance check" if risk_level == "high" else "Monitor closely"
                    })
                    
        return future_anomalies
        
    def generate_early_warning_alerts(self, anomalies: List[Anomaly]) -> List[Dict[str, Any]]:
        """Generate early warning alerts for critical anomalies."""
        alerts = []
        
        for anomaly in anomalies:
            if anomaly.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                alerts.append({
                    "alert_id": f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{anomaly.satellite_id}",
                    "satellite_id": anomaly.satellite_id,
                    "severity": anomaly.severity.value,
                    "message": anomaly.description,
                    "recommendation": anomaly.recommendation,
                    "confidence": anomaly.confidence,
                    "timestamp": anomaly.detected_at.isoformat(),
                    "requires_immediate_action": anomaly.severity == SeverityLevel.CRITICAL
                })
                
        return alerts


# Global anomaly detection system
global_anomaly_detector = SatelliteAnomalyDetector()
predictive_system = PredictiveAnomalySystem()

def detect_satellite_anomalies(data: pd.DataFrame, predictions: Optional[pd.DataFrame] = None) -> List[Anomaly]:
    """Detect anomalies in satellite data."""
    return global_anomaly_detector.detect_anomalies(data, predictions)

def train_anomaly_detector(training_data: pd.DataFrame):
    """Train the global anomaly detector."""
    global_anomaly_detector.fit(training_data)

def get_anomaly_summary(anomalies: List[Anomaly]) -> Dict[str, Any]:
    """Get summary statistics for detected anomalies."""
    if not anomalies:
        return {"total": 0, "by_severity": {}, "by_type": {}}
        
    by_severity = {}
    by_type = {}
    
    for anomaly in anomalies:
        # Count by severity
        severity = anomaly.severity.value
        by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Count by type
        anom_type = anomaly.anomaly_type.value
        by_type[anom_type] = by_type.get(anom_type, 0) + 1
        
    return {
        "total": len(anomalies),
        "by_severity": by_severity,
        "by_type": by_type,
        "most_affected_satellites": [a.satellite_id for a in anomalies[:5]]
    }
