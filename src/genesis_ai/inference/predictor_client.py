import requests
import pandas as pd
from typing import Dict, Any

class GenesisClient:
    """Wrapper to call local or remote GENESIS-AI inference endpoints."""
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url

    def predict_pro(self, df: pd.DataFrame, model_type="gru", seq_len=12, hidden_size=128) -> Dict[str, Any]:
        """
        Call the /predict_pro endpoint with enhanced multi-horizon forecasting and GP uncertainty.
        
        Args:
            df: DataFrame with columns [sat_id, orbit_class, quantity, timestamp, error]
            model_type: "gru" or "transformer"
            seq_len: Sequence length for model input
            hidden_size: Hidden units in the model
            
        Returns:
            Dictionary with predictions and metadata
        """
        # Convert DataFrame to list of records
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "sat_id": str(row["sat_id"]),
                "orbit_class": str(row["orbit_class"]),
                "quantity": str(row["quantity"]),
                "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], 'isoformat') else str(row["timestamp"]),
                "error": float(row["error"])
            })
        
        payload = {
            "model_type": model_type,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "rows": rows,
            "horizons_minutes": [15, 30, 60, 120, 360, 1440],  # 15min to 24h
        }
        
        try:
            r = requests.post(f"{self.base_url}/predict_pro", json=payload, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to call GENESIS-AI API: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Check if the API service is healthy."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Health check failed: {e}")

    def predict_basic(self, df: pd.DataFrame, model_type="gru", seq_len=12, hidden_size=128) -> Dict[str, Any]:
        """
        Call the basic /predict endpoint (deterministic forecasting).
        
        Args:
            df: DataFrame with columns [sat_id, orbit_class, quantity, timestamp, error]
            model_type: "gru" or "transformer"
            seq_len: Sequence length for model input
            hidden_size: Hidden units in the model
            
        Returns:
            Dictionary with predictions and metadata
        """
        # Convert DataFrame to list of records
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "sat_id": str(row["sat_id"]),
                "orbit_class": str(row["orbit_class"]),
                "quantity": str(row["quantity"]),
                "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], 'isoformat') else str(row["timestamp"]),
                "error": float(row["error"])
            })
        
        payload = {
            "model_type": model_type,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "rows": rows,
            "horizons_minutes": [15, 30, 60, 120, 360, 1440],
        }
        
        try:
            r = requests.post(f"{self.base_url}/predict", json=payload, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to call GENESIS-AI API: {e}")
