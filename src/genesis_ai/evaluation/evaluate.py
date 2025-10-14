from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Union

from sklearn.preprocessing import StandardScaler

from genesis_ai.features.engineer import build_feature_frame
from genesis_ai.models.forecast_models import GRUForecaster
from genesis_ai.models.advanced_models import TransformerForecaster
from genesis_ai.evaluation.plots import qq_plot, ks_normality, calibration_hist, ensure_dir

HORIZONS_MIN = [15, 30, 60, 120, 360, 1440]  # minutes

def sliding_windows(X: np.ndarray, y: np.ndarray, seq_len: int, step: int = 1):
    """Yield (x_seq, target_idx) for walk-forward evaluation."""
    for i in range(0, len(X) - seq_len - 1, step):
        yield X[i:i+seq_len], i+seq_len  # predict next after window

def horizon_offsets(minutes: List[int], base_interval_min: int = 15):
    return [max(1, m // base_interval_min) for m in minutes]

def evaluate_series(df: pd.DataFrame, model_type: str = "gru", seq_len: int = 12, hidden_size: int = 128, out_dir: Union[str, Path] = "artifacts/eval"):
    out_dir = ensure_dir(out_dir)
    df = df.sort_values("timestamp")
    df_feat = build_feature_frame(df)

    target = "error"
    ignore = {"sat_id","orbit_class","quantity","timestamp", target}
    feat_cols = [c for c in df_feat.columns if c not in ignore]

    X = df_feat[feat_cols].values.astype(np.float32)
    y = df_feat[target].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Model
    if model_type == "gru":
        model = GRUForecaster(input_size=X.shape[1], hidden_size=hidden_size, num_layers=2, output_size=1)
    elif model_type == "transformer":
        model = TransformerForecaster(input_size=X.shape[1], d_model=hidden_size, nhead=4, num_layers=3, output_size=1)
    else:
        raise ValueError("Unknown model_type")

    # Minimal-fit: train on 70%, evaluate walk-forward on last 30% for clarity
    n = len(X)
    split = int(n * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Simple training loop (single-step next prediction) — lightweight for eval script
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Build sequences for train
    X_seqs, Y_next = [], []
    for i in range(0, len(X_train) - seq_len - 1):
        X_seqs.append(X_train[i:i+seq_len])
        Y_next.append(y_train[i+seq_len])
    X_seqs = torch.tensor(np.stack(X_seqs), dtype=torch.float32)
    Y_next = torch.tensor(np.array(Y_next).reshape(-1,1), dtype=torch.float32)
    dl = DataLoader(TensorDataset(X_seqs, Y_next), batch_size=64, shuffle=True, drop_last=True)

    model.train()
    for _ in range(5):  # few epochs; this is an eval script
        for xb, yb in dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

    # Walk-forward multi-horizon evaluation on test slice
    offs = horizon_offsets(HORIZONS_MIN, base_interval_min=15)
    metrics = {m: {"MAE": None, "RMSE": None} for m in HORIZONS_MIN}
    preds_per_h = {m: [] for m in HORIZONS_MIN}
    trues_per_h = {m: [] for m in HORIZONS_MIN}

    model.eval()
    with torch.no_grad():
        for x_seq, idx in sliding_windows(X_test, y_test, seq_len=seq_len, step=1):
            x_t = torch.tensor(x_seq[None,...], dtype=torch.float32)  # (1,T,F)
            # one-step prediction as base
            base_pred = model(x_t).cpu().numpy().flatten()[0]
            # naive multi-horizon by recursive walk (single-step model) — simple but serviceable
            # for demo: just use same base_pred for all horizons (conservative); extend later to recursive
            for m, k in zip(HORIZONS_MIN, offs):
                tgt_idx = idx + (k - 1)
                if tgt_idx < len(y_test):
                    preds_per_h[m].append(base_pred)
                    trues_per_h[m].append(float(y_test[tgt_idx]))

    # Compute metrics
    for m in HORIZONS_MIN:
        y_true = np.array(trues_per_h[m])
        y_pred = np.array(preds_per_h[m])
        if len(y_true) == 0:
            continue
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        metrics[m]["MAE"] = mae
        metrics[m]["RMSE"] = rmse

    # Residual analysis (aggregate across horizons for now)
    all_res = []
    for m in HORIZONS_MIN:
        y_true = np.array(trues_per_h[m])
        y_pred = np.array(preds_per_h[m])
        if len(y_true):
            all_res.append(y_true - y_pred)
    residuals = np.concatenate(all_res) if all_res else np.array([])

    summary = {}
    if residuals.size > 0:
        summary["normality"] = ks_normality(residuals)
        summary["qq_plot"] = str(qq_plot(residuals, out_dir))
        summary["calibration_hist"] = str(calibration_hist(residuals, out_dir))

    # Save metrics
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_path = Path(out_dir) / "metrics.csv"
    metrics_df.to_csv(metrics_path)

    print("Saved metrics to:", metrics_path)
    if "normality" in summary:
        print("Normality:", summary["normality"])
        print("QQ plot:", summary["qq_plot"])
        print("Calibration hist:", summary["calibration_hist"])

    return metrics_df, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True, help="CSV with columns [sat_id, orbit_class, quantity, timestamp, error]")
    ap.add_argument("--model_type", type=str, default="gru", choices=["gru","transformer"])
    ap.add_argument("--seq_len", type=int, default=12)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="artifacts/eval")
    ap.add_argument("--sat_id", type=str, default=None, help="Optionally filter to a specific satellite id")
    ap.add_argument("--quantity", type=str, default=None, help="clock or ephem")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, parse_dates=["timestamp"])
    if args.sat_id:
        df = df[df["sat_id"] == args.sat_id]
    if args.quantity:
        df = df[df["quantity"] == args.quantity]

    evaluate_series(df, model_type=args.model_type, seq_len=args.seq_len, hidden_size=args.hidden_size, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
