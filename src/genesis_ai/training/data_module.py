from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    """
    Windowed dataset: given features X_t and target y_t, produce sequences of length seq_len to predict y_{t+1}.
    Assumes df is already sorted by timestamp and filtered to one (sat_id, orbit_class, quantity) or pooled with grouping ignored.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 12):
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx: int):
        sl = self.seq_len
        x_seq = self.X[idx: idx + sl]
        y_next = self.y[idx + sl]  # predict next step
        return torch.from_numpy(x_seq), torch.from_numpy(y_next)


class TimeSeriesDataModule:
    """
    Simple DataModule-style wrapper (no Lightning dependency here) that prepares DataLoaders.
    """
    def __init__(self, X_train, y_train, X_val, y_val, batch_size: int = 64, seq_len: int = 12):
        self.train_ds = SequenceDataset(X_train, y_train, seq_len)
        self.val_ds = SequenceDataset(X_val, y_val, seq_len)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
