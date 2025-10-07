import numpy as np
import pandas as pd
from genesis_ai.evaluation.evaluate import evaluate_series

def test_eval_runs(tmp_path):
    ts = pd.date_range("2025-01-01", periods=120, freq="15min")
    df = pd.DataFrame({
        "sat_id": ["S1"]*len(ts),
        "orbit_class": ["MEO"]*len(ts),
        "quantity": ["clock"]*len(ts),
        "timestamp": ts,
        "error": np.random.randn(len(ts))
    })
    m, s = evaluate_series(df, model_type="gru", seq_len=8, hidden_size=32, out_dir=tmp_path)
    assert "MAE" in m.columns
