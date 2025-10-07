import pandas as pd
import numpy as np
from genesis_ai.features.engineer import build_feature_frame

def test_feature_generation_shape():
    ts = pd.date_range("2025-01-01", periods=16, freq="15min")
    df = pd.DataFrame({
        "sat_id": ["S1"]*16,
        "orbit_class": ["MEO"]*16,
        "quantity": ["clock"]*16,
        "timestamp": ts,
        "error": np.random.randn(16)
    })
    out = build_feature_frame(df)
    assert "sin_time" in out.columns
    assert "error_lag_1" in out.columns
    assert "roll_std_8" in out.columns
    assert not out.isna().any().any()
