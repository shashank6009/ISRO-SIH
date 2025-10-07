import pandas as pd
import numpy as np
from genesis_ai.data.loader import load_and_validate, resample_and_impute

def test_load_and_validate_ok(tmp_path):
    csv = tmp_path / "test.csv"
    df = pd.DataFrame({
        "sat_id": ["S1"], "orbit_class": ["MEO"], "quantity": ["clock"],
        "timestamp": ["2025-01-01T00:00:00Z"], "error": [0.1]
    })
    df.to_csv(csv, index=False)
    loaded = load_and_validate(csv)
    assert set(loaded.columns) >= {"sat_id", "timestamp"}

def test_resample_and_impute_adds_mask():
    df = pd.DataFrame({
        "sat_id": ["S1","S1"],
        "orbit_class": ["MEO","MEO"],
        "quantity": ["clock","clock"],
        "timestamp": pd.date_range("2025-01-01", periods=2, freq="30min"),
        "error": [1.0, np.nan],
    })
    out = resample_and_impute(df, interval="15min")
    assert "is_imputed" in out.columns
    assert out["error"].isna().sum() == 0
