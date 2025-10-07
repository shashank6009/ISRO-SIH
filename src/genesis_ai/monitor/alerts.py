import requests
import smtplib
import ssl
import json
import logging
from email.message import EmailMessage
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genesis_ai.db.models import AlertRecord, ForecastRecord, get_engine, get_session

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Detect anomalies in GNSS error predictions."""
    
    def __init__(self, sigma_threshold=3.0, lookback_hours=24):
        self.sigma_threshold = sigma_threshold
        self.lookback_hours = lookback_hours
        self.engine = get_engine()
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect statistical anomalies in prediction data.
        
        Args:
            df: DataFrame with predicted_error column
            
        Returns:
            DataFrame containing only anomalous records
        """
        if len(df) < 3:
            return pd.DataFrame()  # Need minimum data for statistics
        
        # Calculate rolling statistics by satellite and quantity
        anomalies = []
        
        for (sat_id, quantity), group in df.groupby(['sat_id', 'quantity']):
            if len(group) < 3:
                continue
                
            # Calculate statistics
            mean = group["predicted_error"].mean()
            std = group["predicted_error"].std() + 1e-9
            
            # Find outliers
            z_scores = np.abs((group["predicted_error"] - mean) / std)
            outliers = group[z_scores > self.sigma_threshold].copy()
            
            if len(outliers) > 0:
                outliers['z_score'] = z_scores[z_scores > self.sigma_threshold]
                outliers['baseline_mean'] = mean
                outliers['baseline_std'] = std
                anomalies.append(outliers)
        
        if anomalies:
            return pd.concat(anomalies, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def detect_trend_anomalies(self, sat_id: str, quantity: str, window_hours=6) -> Dict[str, Any]:
        """
        Detect trend-based anomalies by comparing recent vs historical performance.
        
        Args:
            sat_id: Satellite identifier
            quantity: 'clock' or 'ephem'
            window_hours: Hours to look back for trend analysis
            
        Returns:
            Dictionary with trend analysis results
        """
        session = get_session(self.engine)
        
        try:
            # Get recent predictions
            cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
            recent_records = session.query(ForecastRecord).filter(
                ForecastRecord.sat_id == sat_id,
                ForecastRecord.quantity == quantity,
                ForecastRecord.created_at >= cutoff_time
            ).all()
            
            if len(recent_records) < 5:
                return {"status": "insufficient_data", "count": len(recent_records)}
            
            # Get historical baseline (last 7 days, excluding recent window)
            historical_cutoff = datetime.utcnow() - timedelta(days=7)
            historical_records = session.query(ForecastRecord).filter(
                ForecastRecord.sat_id == sat_id,
                ForecastRecord.quantity == quantity,
                ForecastRecord.created_at >= historical_cutoff,
                ForecastRecord.created_at < cutoff_time
            ).all()
            
            if len(historical_records) < 10:
                return {"status": "insufficient_baseline", "count": len(historical_records)}
            
            # Calculate statistics
            recent_errors = [r.predicted_error for r in recent_records]
            historical_errors = [r.predicted_error for r in historical_records]
            
            recent_mean = np.mean(recent_errors)
            historical_mean = np.mean(historical_errors)
            historical_std = np.std(historical_errors) + 1e-9
            
            # Trend detection
            trend_z_score = abs(recent_mean - historical_mean) / historical_std
            is_anomalous = trend_z_score > self.sigma_threshold
            
            return {
                "status": "analyzed",
                "sat_id": sat_id,
                "quantity": quantity,
                "recent_mean": recent_mean,
                "historical_mean": historical_mean,
                "trend_z_score": trend_z_score,
                "is_anomalous": is_anomalous,
                "recent_count": len(recent_records),
                "historical_count": len(historical_records),
                "threshold": self.sigma_threshold
            }
            
        finally:
            session.close()

class AlertManager:
    """Manage alerts and notifications for GENESIS-AI."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.engine = get_engine()
        
    def create_alert(self, alert_type: str, severity: str, message: str, 
                    sat_id: Optional[str] = None, details: Optional[Dict] = None) -> int:
        """
        Create a new alert record.
        
        Args:
            alert_type: Type of alert (anomaly, system_error, etc.)
            severity: low, medium, high, critical
            message: Human-readable alert message
            sat_id: Optional satellite ID
            details: Optional additional details as dict
            
        Returns:
            Alert ID
        """
        session = get_session(self.engine)
        
        try:
            alert = AlertRecord(
                alert_type=alert_type,
                severity=severity,
                sat_id=sat_id,
                message=message,
                details=json.dumps(details) if details else None,
                created_at=datetime.utcnow()
            )
            
            session.add(alert)
            session.commit()
            
            logger.info(f"Created {severity} alert: {message}")
            
            # Send notifications based on severity
            if severity in ['high', 'critical']:
                self._send_notifications(alert)
            
            return alert.id
            
        finally:
            session.close()
    
    def _send_notifications(self, alert: AlertRecord):
        """Send notifications for high-priority alerts."""
        try:
            # Send Slack notification if configured
            if 'slack_webhook' in self.config:
                self.send_slack_alert(alert)
            
            # Send email if configured
            if 'email' in self.config:
                self.send_email_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
    
    def send_slack_alert(self, alert: AlertRecord):
        """Send alert to Slack webhook."""
        webhook_url = self.config.get('slack_webhook')
        if not webhook_url:
            return
        
        # Format message for Slack
        color = {
            'low': '#36a64f',
            'medium': '#ff9500', 
            'high': '#ff0000',
            'critical': '#8b0000'
        }.get(alert.severity, '#808080')
        
        payload = {
            "attachments": [{
                "color": color,
                "title": f"🛰️ GENESIS-AI Alert - {alert.severity.upper()}",
                "text": alert.message,
                "fields": [
                    {"title": "Type", "value": alert.alert_type, "short": True},
                    {"title": "Satellite", "value": alert.sat_id or "System", "short": True},
                    {"title": "Time", "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"), "short": True}
                ],
                "footer": "GENESIS-AI Mission Control",
                "ts": int(alert.created_at.timestamp())
            }]
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Slack alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def send_email_alert(self, alert: AlertRecord):
        """Send alert via email."""
        email_config = self.config.get('email', {})
        
        required_fields = ['smtp_server', 'port', 'sender', 'password', 'recipients']
        if not all(field in email_config for field in required_fields):
            logger.warning("Email configuration incomplete, skipping email alert")
            return
        
        try:
            msg = EmailMessage()
            msg["From"] = email_config['sender']
            msg["To"] = ", ".join(email_config['recipients'])
            msg["Subject"] = f"GENESIS-AI Alert: {alert.severity.upper()} - {alert.alert_type}"
            
            body = f"""
GENESIS-AI Mission Control Alert

Severity: {alert.severity.upper()}
Type: {alert.alert_type}
Satellite: {alert.sat_id or 'System'}
Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

Message:
{alert.message}

Details:
{alert.details or 'None'}

---
This is an automated alert from GENESIS-AI Mission Control System.
"""
            msg.set_content(body)
            
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(email_config['smtp_server'], email_config['port'], context=context) as server:
                server.login(email_config['sender'], email_config['password'])
                server.send_message(msg)
            
            logger.info("Email alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def get_recent_alerts(self, hours=24, severity=None) -> List[AlertRecord]:
        """Get recent alerts for dashboard display."""
        session = get_session(self.engine)
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            query = session.query(AlertRecord).filter(AlertRecord.created_at >= cutoff_time)
            
            if severity:
                query = query.filter(AlertRecord.severity == severity)
            
            return query.order_by(AlertRecord.created_at.desc()).all()
            
        finally:
            session.close()
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str):
        """Acknowledge an alert."""
        session = get_session(self.engine)
        
        try:
            alert = session.query(AlertRecord).filter(AlertRecord.id == alert_id).first()
            if alert:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by
                session.commit()
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            
        finally:
            session.close()

class MonitoringService:
    """Main monitoring service that coordinates anomaly detection and alerting."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.detector = AnomalyDetector()
        self.alert_manager = AlertManager(config)
        
    def monitor_predictions(self, predictions_df: pd.DataFrame):
        """
        Monitor a batch of predictions for anomalies.
        
        Args:
            predictions_df: DataFrame with prediction results
        """
        try:
            # Detect statistical anomalies
            anomalies = self.detector.detect_anomalies(predictions_df)
            
            if len(anomalies) > 0:
                for _, anomaly in anomalies.iterrows():
                    message = (
                        f"Anomalous prediction detected for {anomaly['sat_id']} "
                        f"({anomaly['quantity']}): {anomaly['predicted_error']:.4f} "
                        f"(z-score: {anomaly.get('z_score', 'N/A'):.2f})"
                    )
                    
                    severity = 'high' if anomaly.get('z_score', 0) > 5 else 'medium'
                    
                    self.alert_manager.create_alert(
                        alert_type='anomaly',
                        severity=severity,
                        message=message,
                        sat_id=anomaly['sat_id'],
                        details={
                            'quantity': anomaly['quantity'],
                            'predicted_error': float(anomaly['predicted_error']),
                            'z_score': float(anomaly.get('z_score', 0)),
                            'baseline_mean': float(anomaly.get('baseline_mean', 0)),
                            'baseline_std': float(anomaly.get('baseline_std', 0))
                        }
                    )
            
            # Check for trend anomalies
            satellites = predictions_df[['sat_id', 'quantity']].drop_duplicates()
            for _, row in satellites.iterrows():
                trend_analysis = self.detector.detect_trend_anomalies(row['sat_id'], row['quantity'])
                
                if trend_analysis.get('is_anomalous', False):
                    message = (
                        f"Trend anomaly detected for {row['sat_id']} ({row['quantity']}): "
                        f"Recent mean {trend_analysis['recent_mean']:.4f} vs "
                        f"historical {trend_analysis['historical_mean']:.4f} "
                        f"(z-score: {trend_analysis['trend_z_score']:.2f})"
                    )
                    
                    self.alert_manager.create_alert(
                        alert_type='trend_anomaly',
                        severity='medium',
                        message=message,
                        sat_id=row['sat_id'],
                        details=trend_analysis
                    )
                    
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            self.alert_manager.create_alert(
                alert_type='system_error',
                severity='high',
                message=f"Monitoring system error: {str(e)[:200]}",
                details={'error': str(e)}
            )

# Utility functions for external use
def detect_anomalies(df: pd.DataFrame, sigma_threshold: float = 3.0) -> pd.DataFrame:
    """Standalone anomaly detection function."""
    detector = AnomalyDetector(sigma_threshold=sigma_threshold)
    return detector.detect_anomalies(df)

def send_slack(webhook_url: str, text: str):
    """Simple Slack notification function."""
    try:
        response = requests.post(webhook_url, json={"text": text}, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Slack notification failed: {e}")

def send_email(smtp_server: str, port: int, sender: str, password: str, 
               recipient: str, subject: str, body: str):
    """Simple email notification function."""
    try:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = sender, recipient, subject
        msg.set_content(body)
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender, password)
            server.send_message(msg)
    except Exception as e:
        logger.error(f"Email notification failed: {e}")

if __name__ == "__main__":
    # Test the monitoring system
    import numpy as np
    
    # Create test data with some anomalies
    np.random.seed(42)
    test_data = pd.DataFrame({
        'sat_id': ['IRNSS-1A'] * 20,
        'quantity': ['clock'] * 20,
        'predicted_error': np.concatenate([
            np.random.normal(0.05, 0.01, 18),  # Normal data
            [0.15, 0.20]  # Anomalies
        ])
    })
    
    print("Testing anomaly detection...")
    detector = AnomalyDetector()
    anomalies = detector.detect_anomalies(test_data)
    print(f"Detected {len(anomalies)} anomalies")
    
    if len(anomalies) > 0:
        print("Anomalous records:")
        print(anomalies[['sat_id', 'predicted_error', 'z_score']])
    
    print("Monitoring system test completed.")
