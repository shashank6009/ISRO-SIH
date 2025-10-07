from typing import Literal, Dict
import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl

def normality_regularizer(residuals: torch.Tensor) -> torch.Tensor:
    """
    Penalize deviation from Gaussian: skew^2 + (kurtosis-3)^2.
    residuals: (batch, 1)
    """
    x = residuals.view(-1)
    if x.numel() < 4:
        return torch.tensor(0.0, device=x.device)
    mean = torch.mean(x)
    std = torch.std(x) + 1e-6
    z = (x - mean) / std
    skew = torch.mean(z**3)
    kurt = torch.mean(z**4)
    return skew.pow(2) + (kurt - 3.0).pow(2)

class ForecastLightning(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3, normality_weight: float = 0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.MSELoss()
        self.normality_weight = normality_weight
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, T, F), y: (B,1)
        y_hat = self(x)
        mse = self.criterion(y_hat, y)
        res = y - y_hat
        norm_pen = normality_regularizer(res)
        loss = mse + self.normality_weight * norm_pen
        self.log_dict({"train_mse": mse, "train_norm_pen": norm_pen, "train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse = self.criterion(y_hat, y)
        res = y - y_hat
        norm_pen = normality_regularizer(res)
        loss = mse + self.normality_weight * norm_pen
        self.log_dict({"val_mse": mse, "val_norm_pen": norm_pen, "val_loss": loss}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
