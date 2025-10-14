"""
Normality-Aware Loss Functions for GNSS Error Prediction
Implements custom loss functions that optimize for both accuracy and normal distribution of errors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from typing import Tuple, Dict

class NormalityLoss(nn.Module):
    """Loss function that penalizes deviations from normal distribution"""
    
    def __init__(self, normality_weight: float = 0.7, method: str = 'ks'):
        super().__init__()
        self.normality_weight = normality_weight
        self.method = method
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Calculate residuals
        residuals = (predictions - targets).detach().cpu().numpy().flatten()
        
        # Normality penalty
        normality_penalty = self._compute_normality_penalty(residuals)
        
        # Combined loss
        total_loss = mse_loss + self.normality_weight * normality_penalty
        
        return total_loss
    
    def _compute_normality_penalty(self, residuals: np.ndarray) -> torch.Tensor:
        """Compute penalty based on deviation from normal distribution"""
        if len(residuals) < 8:  # Need minimum samples for statistical tests
            return torch.tensor(0.0)
        
        # Standardize residuals
        residuals_std = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8)
        
        if self.method == 'ks':
            # Kolmogorov-Smirnov test
            ks_stat, _ = stats.kstest(residuals_std, 'norm')
            penalty = torch.tensor(ks_stat, dtype=torch.float32)
            
        elif self.method == 'ad':
            # Anderson-Darling test
            ad_stat, _, _ = stats.anderson(residuals_std, dist='norm')
            penalty = torch.tensor(ad_stat / 10.0, dtype=torch.float32)  # Scale down
            
        elif self.method == 'sw':
            # Shapiro-Wilk test (for smaller samples)
            if len(residuals_std) <= 5000:
                sw_stat, _ = stats.shapiro(residuals_std)
                penalty = torch.tensor(1.0 - sw_stat, dtype=torch.float32)
            else:
                # Fall back to KS for large samples
                ks_stat, _ = stats.kstest(residuals_std, 'norm')
                penalty = torch.tensor(ks_stat, dtype=torch.float32)
                
        elif self.method == 'skew_kurt':
            # Skewness and kurtosis penalty
            skewness = stats.skew(residuals_std)
            kurtosis = stats.kurtosis(residuals_std)
            penalty = torch.tensor(abs(skewness) + abs(kurtosis) / 3.0, dtype=torch.float32)
        
        else:
            raise ValueError(f"Unknown normality method: {self.method}")
        
        return penalty

class MultiHorizonNormalityLoss(nn.Module):
    """Loss function for multiple prediction horizons with normality constraints"""
    
    def __init__(self, horizon_weights: Dict[str, float], normality_weight: float = 0.1):
        super().__init__()
        self.horizon_weights = horizon_weights
        self.normality_losses = nn.ModuleDict({
            horizon: NormalityLoss(normality_weight, method='ks')
            for horizon in horizon_weights.keys()
        })
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        total_loss = torch.tensor(0.0)
        horizon_losses = {}
        
        for horizon in self.horizon_weights.keys():
            if horizon in predictions and horizon in targets:
                horizon_loss = self.normality_losses[horizon](predictions[horizon], targets[horizon])
                weighted_loss = self.horizon_weights[horizon] * horizon_loss
                
                total_loss += weighted_loss
                horizon_losses[horizon] = horizon_loss
        
        return total_loss, horizon_losses

class AdaptiveNormalityLoss(nn.Module):
    """Adaptive loss that adjusts normality weight based on training progress"""
    
    def __init__(self, initial_normality_weight: float = 0.01, 
                 max_normality_weight: float = 0.2, warmup_epochs: int = 50):
        super().__init__()
        self.initial_weight = initial_normality_weight
        self.max_weight = max_normality_weight
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Adaptive normality weight
        progress = min(self.current_epoch / self.warmup_epochs, 1.0)
        current_weight = self.initial_weight + progress * (self.max_weight - self.initial_weight)
        
        # Calculate residuals and normality penalty
        residuals = (predictions - targets).detach().cpu().numpy().flatten()
        normality_penalty = self._compute_combined_normality_penalty(residuals)
        
        total_loss = mse_loss + current_weight * normality_penalty
        
        return total_loss
    
    def _compute_combined_normality_penalty(self, residuals: np.ndarray) -> torch.Tensor:
        """Combine multiple normality tests for robust penalty"""
        if len(residuals) < 8:
            return torch.tensor(0.0)
        
        residuals_std = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8)
        
        penalties = []
        
        # Kolmogorov-Smirnov test
        try:
            ks_stat, _ = stats.kstest(residuals_std, 'norm')
            penalties.append(ks_stat)
        except:
            pass
        
        # Skewness and kurtosis
        try:
            skewness = abs(stats.skew(residuals_std))
            kurtosis = abs(stats.kurtosis(residuals_std)) / 3.0
            penalties.extend([skewness, kurtosis])
        except:
            pass
        
        # Jarque-Bera test
        try:
            jb_stat, _ = stats.jarque_bera(residuals_std)
            penalties.append(jb_stat / 100.0)  # Scale down
        except:
            pass
        
        if penalties:
            combined_penalty = torch.tensor(np.mean(penalties), dtype=torch.float32)
        else:
            combined_penalty = torch.tensor(0.0)
        
        return combined_penalty
    
    def step_epoch(self):
        """Call this at the end of each epoch"""
        self.current_epoch += 1

class UncertaintyAwareNormalityLoss(nn.Module):
    """Loss function that incorporates prediction uncertainty with normality constraints"""
    
    def __init__(self, normality_weight: float = 0.1, uncertainty_weight: float = 0.05):
        super().__init__()
        self.normality_weight = normality_weight
        self.uncertainty_weight = uncertainty_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                uncertainties: torch.Tensor) -> torch.Tensor:
        
        # Heteroscedastic loss (uncertainty-weighted MSE)
        precision = 1.0 / (uncertainties + 1e-8)
        mse_loss = torch.mean(precision * (predictions - targets)**2 + torch.log(uncertainties + 1e-8))
        
        # Uncertainty regularization (prevent overconfidence)
        uncertainty_reg = torch.mean(uncertainties)
        
        # Normality penalty on standardized residuals
        residuals = (predictions - targets).detach().cpu().numpy().flatten()
        normality_penalty = self._compute_normality_penalty(residuals)
        
        total_loss = (mse_loss + 
                     self.uncertainty_weight * uncertainty_reg + 
                     self.normality_weight * normality_penalty)
        
        return total_loss
    
    def _compute_normality_penalty(self, residuals: np.ndarray) -> torch.Tensor:
        """Compute normality penalty using multiple tests"""
        if len(residuals) < 8:
            return torch.tensor(0.0)
        
        residuals_std = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8)
        
        # Use KS test as primary normality measure
        try:
            ks_stat, _ = stats.kstest(residuals_std, 'norm')
            return torch.tensor(ks_stat, dtype=torch.float32)
        except:
            return torch.tensor(0.0)

class DistributionMatchingLoss(nn.Module):
    """Loss function that explicitly matches target error distribution"""
    
    def __init__(self, target_distribution: str = 'normal', 
                 distribution_weight: float = 0.15):
        super().__init__()
        self.target_distribution = target_distribution
        self.distribution_weight = distribution_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Distribution matching loss
        residuals = predictions - targets
        dist_loss = self._compute_distribution_loss(residuals)
        
        total_loss = mse_loss + self.distribution_weight * dist_loss
        
        return total_loss
    
    def _compute_distribution_loss(self, residuals: torch.Tensor) -> torch.Tensor:
        """Compute loss based on distribution matching"""
        if self.target_distribution == 'normal':
            # Encourage zero mean and unit variance
            mean_penalty = torch.abs(torch.mean(residuals))
            var_penalty = torch.abs(torch.var(residuals) - 1.0)
            
            # Encourage symmetric distribution
            residuals_sorted = torch.sort(residuals.flatten())[0]
            n = len(residuals_sorted)
            if n > 4:
                # Compare quantiles to normal distribution
                normal_quantiles = torch.tensor([
                    stats.norm.ppf((i + 1) / (n + 1)) for i in range(n)
                ], dtype=residuals.dtype, device=residuals.device)
                
                quantile_loss = F.mse_loss(residuals_sorted, normal_quantiles * torch.std(residuals))
            else:
                quantile_loss = torch.tensor(0.0)
            
            return mean_penalty + var_penalty + quantile_loss
        
        else:
            raise ValueError(f"Unsupported target distribution: {self.target_distribution}")

def create_competition_loss(loss_type: str = 'adaptive', **kwargs) -> nn.Module:
    """Factory function to create competition-optimized loss functions"""
    
    if loss_type == 'normality':
        return NormalityLoss(**kwargs)
    elif loss_type == 'multi_horizon':
        return MultiHorizonNormalityLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveNormalityLoss(**kwargs)
    elif loss_type == 'uncertainty':
        return UncertaintyAwareNormalityLoss(**kwargs)
    elif loss_type == 'distribution':
        return DistributionMatchingLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
