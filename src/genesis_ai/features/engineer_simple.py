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

def add_lags_safe(df: pd.DataFrame, lags=(1, 2)) -> pd.DataFrame:
    """Add lag features for the error column with safe handling."""
    df = df.copy()
    for lag in lags:
        df[f"error_lag_{lag}"] = df.groupby(["sat_id", "orbit_class", "quantity"])["error"].shift(lag)
        # Fill NaN with the mean error for that satellite
        df[f"error_lag_{lag}"] = df[f"error_lag_{lag}"].fillna(df.groupby(["sat_id", "orbit_class", "quantity"])["error"].transform('mean'))
        # If still NaN (single row groups), fill with overall mean
        df[f"error_lag_{lag}"] = df[f"error_lag_{lag}"].fillna(df["error"].mean())
    return df

def add_rolling_stats_safe(df: pd.DataFrame, windows=(4, 8)) -> pd.DataFrame:
    """Add rolling mean and std for error with safe handling."""
    df = df.copy()
    for w in windows:
        grp = df.groupby(["sat_id", "orbit_class", "quantity"])["error"]
        df[f"roll_mean_{w}"] = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"] = grp.transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0))
    return df

def add_satellite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add satellite-specific features."""
    df = df.copy()
    
    # Orbit class encoding
    orbit_map = {"LEO": 0, "MEO": 1, "GEO/GSO": 2}
    df["orbit_encoded"] = df["orbit_class"].map(orbit_map).fillna(0)
    
    # Quantity encoding
    quantity_map = {"clock": 0, "ephem": 1}
    df["quantity_encoded"] = df["quantity"].map(quantity_map).fillna(0)
    
    # Satellite ID hash (for model diversity)
    df["sat_hash"] = df["sat_id"].apply(lambda x: hash(x) % 1000 / 1000.0)
    
    return df

def build_feature_frame_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Safe feature pipeline that doesn't drop rows unnecessarily."""
    if len(df) == 0:
        return df
        
    df = df.copy()
    df = add_time_encodings(df)
    df = add_satellite_features(df)
    
    # Only add lags and rolling stats if we have enough data
    if len(df) >= 2:
        df = add_lags_safe(df)
    else:
        # Add dummy lag features for consistency
        df["error_lag_1"] = df["error"]
        df["error_lag_2"] = df["error"]
    
    if len(df) >= 4:
        df = add_rolling_stats_safe(df)
    else:
        # Add dummy rolling features for consistency
        df["roll_mean_4"] = df["error"]
        df["roll_std_4"] = 0.0
        df["roll_mean_8"] = df["error"]
        df["roll_std_8"] = 0.0
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    return df

if __name__ == "__main__":
    # Test with minimal data
    ts = pd.date_range("2025-01-01", periods=3, freq="15min")
    sample = pd.DataFrame({
        "sat_id": ["S1"]*3,
        "orbit_class": ["MEO"]*3,
        "quantity": ["clock"]*3,
        "timestamp": ts,
        "error": [0.1, 0.2, 0.15]
    })
    feats = build_feature_frame_safe(sample)
    print("Generated columns:", feats.columns.tolist())
    print("Shape:", feats.shape)
    print("No NaN values:", not feats.isnull().any().any())
