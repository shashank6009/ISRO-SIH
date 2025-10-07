from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def qq_plot(residuals: np.ndarray, out_dir: Union[str, Path], title: str = "Residuals QQ-Plot"):
    out_dir = ensure_dir(out_dir)
    fig, ax = plt.subplots()
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(title)
    fig.tight_layout()
    path = Path(out_dir) / "qq_plot.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path

def ks_normality(residuals: np.ndarray):
    mu, sigma = np.mean(residuals), np.std(residuals) + 1e-9
    z = (residuals - mu) / sigma
    D, p = stats.kstest(z, "norm")
    return {"mu": mu, "sigma": sigma, "ks_D": float(D), "ks_p": float(p)}

def calibration_hist(residuals: np.ndarray, out_dir: Union[str, Path], title: str = "Residuals Calibration"):
    out_dir = ensure_dir(out_dir)
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=30, density=True, alpha=0.6, label="Empirical")
    mu, sigma = np.mean(residuals), np.std(residuals) + 1e-9
    xs = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
    ax.plot(xs, stats.norm.pdf(xs, loc=mu, scale=sigma), label="Fitted Normal")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path = Path(out_dir) / "calibration_hist.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
