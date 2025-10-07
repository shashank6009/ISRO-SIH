import numpy as np
import pandas as pd
from genesis_ai.training.train import build_Xy
from genesis_ai.features.engineer import build_feature_frame

def test_build_Xy_columns():
    ts = pd.date_range("2025-01-01", periods=40, freq="15min")
    df = pd.DataFrame({
        "sat_id": ["S1"]*40,
        "orbit_class": ["MEO"]*40,
        "quantity": ["clock"]*40,
        "timestamp": ts,
        "error": np.random.randn(40)
    })
    df = build_feature_frame(df)
    X, y, feats = build_Xy(df)
    assert X.shape[0] == y.shape[0]
    assert len(feats) > 3
