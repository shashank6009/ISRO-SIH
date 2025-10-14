import argparse
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from genesis_ai.features.engineer import build_feature_frame
from genesis_ai.training.data_module import TimeSeriesDataModule
from genesis_ai.training.lightning_module import ForecastLightning
from genesis_ai.models.forecast_models import GRUForecaster
from genesis_ai.models.advanced_models import TransformerForecaster

def build_Xy(df: pd.DataFrame):
    # Select numeric features except target
    target = "error"
    ignore = {"sat_id","orbit_class","quantity","timestamp", target}
    feat_cols = [c for c in df.columns if c not in ignore]
    X = df[feat_cols].values
    y = df[target].values
    return X, y, feat_cols

def main(args):
    # Expect preprocessed CSV (resampled) — or use a mock path for now
    df = pd.read_csv(args.input_csv, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = build_feature_frame(df)
    X, y, feat_cols = build_Xy(df)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    dm = TimeSeriesDataModule(X_train, y_train, X_val, y_val, batch_size=args.batch_size, seq_len=args.seq_len)

    # --- Model selection ---
    if args.model_type == "gru":
        model = GRUForecaster(
            input_size=X.shape[1],
            hidden_size=args.hidden_size,
            num_layers=2,
            output_size=1,
        )
    elif args.model_type == "transformer":
        model = TransformerForecaster(
            input_size=X.shape[1],
            d_model=args.hidden_size,
            nhead=4,
            num_layers=3,
            output_size=1,
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    lit = ForecastLightning(model=model, lr=args.lr, normality_weight=args.normality_weight)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_model_summary=True,
    )
    trainer.fit(lit, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    print("Training complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, required=True, help="Path to resampled CSV with columns incl. 'error' and 'timestamp'")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=12)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--normality_weight", type=float, default=0.1)
    p.add_argument("--model_type", type=str, default="gru", choices=["gru", "transformer"])
    args = p.parse_args()
    main(args)
