import pytest
import pandas as pd

REQUIRED_COLUMNS = ["sat_id", "orbit_class", "quantity", "timestamp", "error"]

def validate_schema(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def test_validate_schema_passes():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    validate_schema(df)

def test_validate_schema_fails():
    df = pd.DataFrame(columns=["sat_id", "timestamp"])
    with pytest.raises(ValueError):
        validate_schema(df)
