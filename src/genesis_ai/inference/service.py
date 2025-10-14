from __future__ import annotations
from typing import List, Literal, Optional
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import torch

try:
    from genesis_ai.features.engineer_simple import build_feature_frame_safe as build_feature_frame
except ImportError:
    from genesis_ai.features.engineer import build_feature_frame
from genesis_ai.models.forecast_models import GRUForecaster
from genesis_ai.models.advanced_models import TransformerForecaster
from genesis_ai.models.hybrid_model import HybridWithGP

# ---------- Pydantic Schemas ----------
class Row(BaseModel):
    sat_id: str
    orbit_class: Literal["GEO/GSO", "MEO", "LEO"] = "MEO"
    quantity: Literal["clock", "ephem"]
    timestamp: datetime
    error: float

class PredictRequest(BaseModel):
    model_type: Literal["gru", "transformer"] = "gru"
    seq_len: int = 12
    hidden_size: int = 128
    horizons_minutes: List[int] = Field(default_factory=lambda: [15, 30, 60, 120, 360, 1440])
    rows: List[Row]

class Prediction(BaseModel):
    timestamp: datetime
    sat_id: str
    orbit_class: str
    quantity: str
    predicted_error: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

class PredictResponse(BaseModel):
    predictions: List[Prediction]
    info: dict

# ---------- App ----------
app = FastAPI(title="GENESIS-AI Inference Service", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "service": "genesis-ai", "version": "0.1.0"}

# Helper: simple model factory (weights loading can be added later)
def make_model(model_type: str, input_size: int, hidden_size: int):
    if model_type == "gru":
        return GRUForecaster(input_size=input_size, hidden_size=hidden_size, num_layers=2, output_size=1)
    elif model_type == "transformer":
        return TransformerForecaster(input_size=input_size, d_model=hidden_size, nhead=4, num_layers=3, output_size=1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # 1) Build DataFrame from rows and sort
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2) Feature frame
    feat_df = build_feature_frame(df)
    target = "error"
    ignore = {"sat_id","orbit_class","quantity","timestamp", target}
    feat_cols = [c for c in feat_df.columns if c not in ignore]

    # Must have enough history for seq_len
    if len(feat_df) <= req.seq_len:
        raise ValueError(f"Need > seq_len={req.seq_len} rows after feature build; got {len(feat_df)}")

    # 3) Scale features (fit-on-context; in prod, replace with persisted scaler)
    X = feat_df[feat_cols].values.astype(np.float32)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # 4) Build one window for last seq_len steps
    x_seq = torch.tensor(X[-req.seq_len:], dtype=torch.float32).unsqueeze(0)  # (1,T,F)
    model = make_model(req.model_type, input_size=X.shape[1], hidden_size=req.hidden_size)
    model.eval()
    with torch.no_grad():
        base_pred = float(model(x_seq).cpu().numpy().flatten()[0])

    # 5) Produce horizon timestamps by adding minutes to the last known timestamp
    t0 = pd.to_datetime(df["timestamp"].iloc[-1])
    preds: List[Prediction] = []
    for m in req.horizons_minutes:
        t_pred = (t0 + pd.Timedelta(minutes=int(m))).to_pydatetime()
        preds.append(Prediction(
            timestamp=t_pred,
            sat_id=str(df["sat_id"].iloc[-1]),
            orbit_class=str(df["orbit_class"].iloc[-1]),
            quantity=str(df["quantity"].iloc[-1]),
            predicted_error=base_pred,
            lower_bound=None,  # will be filled when GP head is integrated in /predict_pro
            upper_bound=None
        ))

    return PredictResponse(
        predictions=preds,
        info={
            "model_type": req.model_type,
            "seq_len": req.seq_len,
            "hidden_size": req.hidden_size,
            "horizons_minutes": req.horizons_minutes,
            "note": "Current horizon outputs reuse next-step prediction; multi-step recursion + GP UQ will be added in next prompts."
        }
    )

@app.post("/predict_pro", response_model=PredictResponse)
def predict_pro(req: PredictRequest):
    """
    Enhanced endpoint: recursive multi-step forecasting + Gaussian Process uncertainty.
    """
    df = pd.DataFrame([r.model_dump() for r in req.rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    feat_df = build_feature_frame(df)
    target = "error"
    ignore = {"sat_id","orbit_class","quantity","timestamp", target}
    feat_cols = [c for c in feat_df.columns if c not in ignore]

    if len(feat_df) <= req.seq_len:
        raise ValueError(f"Need > seq_len={req.seq_len} rows after feature build; got {len(feat_df)}")

    from sklearn.preprocessing import StandardScaler
    X = feat_df[feat_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    x_seq = torch.tensor(X[-req.seq_len:], dtype=torch.float32).unsqueeze(0)
    model = make_model(req.model_type, input_size=X.shape[1], hidden_size=req.hidden_size)
    model.eval()

    preds = []
    base_time = pd.to_datetime(df["timestamp"].iloc[-1])
    last_values = df["error"].values.tolist()

    # --- recursive prediction ---
    with torch.no_grad():
        for m in req.horizons_minutes:
            x_input = torch.tensor(X[-req.seq_len:], dtype=torch.float32).unsqueeze(0)
            y_hat = model(x_input).cpu().numpy().flatten()[0]
            preds.append(y_hat)
            # roll forward (simplified recursion)
            last_values.append(y_hat)
            new_row = feat_df.iloc[-1:].copy()
            new_row["error"] = y_hat
            feat_df = pd.concat([feat_df, new_row], ignore_index=True)
            X = feat_df[feat_cols].values.astype(np.float32)
            X = scaler.transform(X).astype(np.float32)

    preds_t = torch.tensor(preds).unsqueeze(1)
    y_train = torch.tensor(df["error"].values[-len(preds):]).unsqueeze(1)

    gp = HybridWithGP()
    gp.fit_gp(preds_t, y_train)
    mean, var = gp.predict(preds_t)

    t0 = base_time
    out_preds = []
    for i, m in enumerate(req.horizons_minutes):
        ts_pred = (t0 + pd.Timedelta(minutes=int(m))).to_pydatetime()
        std = float(var[i].sqrt().item()) if var is not None else 0.0
        mu = float(mean[i].item())
        out_preds.append(Prediction(
            timestamp=ts_pred,
            sat_id=str(df["sat_id"].iloc[-1]),
            orbit_class=str(df["orbit_class"].iloc[-1]),
            quantity=str(df["quantity"].iloc[-1]),
            predicted_error=mu,
            lower_bound=mu - 2*std,
            upper_bound=mu + 2*std
        ))

    return PredictResponse(
        predictions=out_preds,
        info={
            "model_type": req.model_type,
            "seq_len": req.seq_len,
            "hidden_size": req.hidden_size,
            "gp_confidence": "±2σ bounds",
            "note": "Recursive multi-step forecast with GP uncertainty calibration."
        }
    )
