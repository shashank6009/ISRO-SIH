import pandas as pd
import numpy as np

def add_time_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical encodings for time of day and day of week."""
    df = df.copy()
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["sin_time"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["cos_time"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df

def add_lags(df: pd.DataFrame, lags=(1, 2, 4, 8)) -> pd.DataFrame:
    """Add lag features for the error column (in time steps)."""
    df = df.copy()
    for lag in lags:
        df[f"error_lag_{lag}"] = df.groupby(["sat_id", "orbit_class", "quantity"])["error"].shift(lag)
    return df

def add_rolling_stats(df: pd.DataFrame, windows=(4, 8, 24)) -> pd.DataFrame:
    """Add rolling mean and std for error over given windows (in 15-min steps)."""
    df = df.copy()
    for w in windows:
        grp = df.groupby(["sat_id", "orbit_class", "quantity"])["error"]
        df[f"roll_mean_{w}"] = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"] = grp.transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0))
    return df

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline."""
    df = add_time_encodings(df)
    df = add_lags(df)
    df = add_rolling_stats(df)
    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    # Simple smoke test
    ts = pd.date_range("2025-01-01", periods=10, freq="15min")
    sample = pd.DataFrame({
        "sat_id": ["S1"]*10,
        "orbit_class": ["MEO"]*10,
        "quantity": ["clock"]*10,
        "timestamp": ts,
        "error": np.random.randn(10)
    })
    feats = build_feature_frame(sample)
    print("Generated columns:", feats.columns.tolist())
