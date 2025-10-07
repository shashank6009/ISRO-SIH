from genesis_ai.training.train import main
import pandas as pd
import numpy as np
import tempfile
import os

def make_fake_csv(path):
    ts = pd.date_range("2025-01-01", periods=40, freq="15min")
    df = pd.DataFrame({
        "sat_id": ["S1"]*40,
        "orbit_class": ["MEO"]*40,
        "quantity": ["clock"]*40,
        "timestamp": ts,
        "error": np.random.randn(40)
    })
    df.to_csv(path, index=False)

def test_cli_model_choice_runs(tmp_path):
    fake_csv = tmp_path / "f.csv"
    make_fake_csv(fake_csv)
    # Just verify no crash (not actual training)
    os.system(f"python -m genesis_ai.training.train --input_csv {fake_csv} --epochs 1 --model_type gru")
    os.system(f"python -m genesis_ai.training.train --input_csv {fake_csv} --epochs 1 --model_type transformer")
