"""
Competition-Optimized Loss Functions for GNSS Error Prediction
Prioritizes normal distribution (70%) over accuracy (30%) as per competition requirements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional, List, Any
import warnings

class CompetitionLoss(nn.Module):
    """
    Competition-optimized loss function that prioritizes normality over accuracy
    - Normality: 70% weight (primary evaluation criterion)
    - Accuracy: 30% weight (secondary criterion)
    """
    
    def __init__(self, normality_weight: float = 0.7, accuracy_weight: float = 0.3, 
                 normality_methods: List[str] = ['ks', 'anderson', 'shapiro']):
        super().__init__()
        self.normality_weight = normality_weight
        self.accuracy_weight = accuracy_weight
        self.normality_methods = normality_methods
        
        # Ensure weights sum to 1
        total_weight = normality_weight + accuracy_weight
        self.normality_weight = normality_weight / total_weight
        self.accuracy_weight = accuracy_weight / total_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute competition loss with detailed breakdown
        
        Returns:
            loss: Combined loss tensor
            metrics: Dictionary with loss component breakdown
        """
        # Accuracy loss (MSE)
        accuracy_loss = F.mse_loss(predictions, targets)
        
        # Calculate residuals
        residuals = (predictions - targets).detach().cpu().numpy().flatten()
        
        # Normality loss (multiple methods)
        normality_loss, normality_metrics = self._compute_comprehensive_normality_loss(residuals)
        
        # Combined loss
        total_loss = (self.accuracy_weight * accuracy_loss + 
                     self.normality_weight * normality_loss)
        
        # Detailed metrics
        metrics = {
            'total_loss': total_loss.item(),
            'accuracy_loss': accuracy_loss.item(),
            'normality_loss': normality_loss.item(),
            'accuracy_weight': self.accuracy_weight,
            'normality_weight': self.normality_weight,
            **normality_metrics
        }
        
        return total_loss, metrics
    
    def _compute_comprehensive_normality_loss(self, residuals: np.ndarray) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute normality loss using multiple statistical tests"""
        
        if len(residuals) < 8:  # Need minimum samples for statistical tests
            return torch.tensor(1.0, requires_grad=True), {'normality_warning': 'insufficient_samples'}
        
        normality_penalties = []
        metrics = {}
        
        # Kolmogorov-Smirnov test
        if 'ks' in self.normality_methods:
            try:
                # Standardize residuals
                standardized = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8)
                ks_stat, ks_p = stats.kstest(standardized, 'norm')
                ks_penalty = ks_stat  # Higher stat = worse normality
                normality_penalties.append(ks_penalty)
                metrics['ks_statistic'] = ks_stat
                metrics['ks_p_value'] = ks_p
            except Exception as e:
                warnings.warn(f"KS test failed: {e}")
                normality_penalties.append(1.0)
        
        # Anderson-Darling test
        if 'anderson' in self.normality_methods:
            try:
                standardized = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8)
                ad_result = stats.anderson(standardized, dist='norm')
                # Normalize AD statistic (typically ranges 0-10+)
                ad_penalty = min(ad_result.statistic / 10.0, 1.0)
                normality_penalties.append(ad_penalty)
                metrics['anderson_statistic'] = ad_result.statistic
                metrics['anderson_critical_values'] = ad_result.critical_values.tolist()
            except Exception as e:
                warnings.warn(f"Anderson-Darling test failed: {e}")
                normality_penalties.append(1.0)
        
        # Shapiro-Wilk test (for smaller samples)
        if 'shapiro' in self.normality_methods and len(residuals) <= 5000:
            try:
                standardized = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8)
                sw_stat, sw_p = stats.shapiro(standardized[:5000])  # Limit sample size
                sw_penalty = 1.0 - sw_stat  # Higher stat = better normality, so invert
                normality_penalties.append(sw_penalty)
                metrics['shapiro_statistic'] = sw_stat
                metrics['shapiro_p_value'] = sw_p
            except Exception as e:
                warnings.warn(f"Shapiro-Wilk test failed: {e}")
                normality_penalties.append(1.0)
        
        # Histogram entropy penalty (custom normality measure)
        if 'entropy' in self.normality_methods:
            try:
                entropy_penalty = self._compute_histogram_entropy_penalty(residuals)
                normality_penalties.append(entropy_penalty)
                metrics['histogram_entropy'] = entropy_penalty
            except Exception as e:
                warnings.warn(f"Entropy computation failed: {e}")
                normality_penalties.append(1.0)
        
        # Average normality penalty
        if normality_penalties:
            avg_penalty = np.mean(normality_penalties)
            metrics['avg_normality_penalty'] = avg_penalty
        else:
            avg_penalty = 1.0
            metrics['normality_warning'] = 'all_methods_failed'
        
        return torch.tensor(avg_penalty, requires_grad=True), metrics
    
    def _compute_histogram_entropy_penalty(self, residuals: np.ndarray) -> float:
        """Compute normality penalty based on histogram entropy"""
        try:
            # Create histogram
            hist, _ = np.histogram(residuals, bins=min(50, len(residuals) // 10), density=True)
            hist = hist + 1e-8  # Avoid log(0)
            
            # Compute entropy
            entropy = -np.sum(hist * np.log(hist))
            
            # Expected entropy for normal distribution (approximate)
            # Normal distribution has high entropy, so we want to maximize entropy
            # Penalty = 1 - normalized_entropy
            max_entropy = np.log(len(hist))  # Maximum possible entropy
            normalized_entropy = entropy / max_entropy
            
            # Return penalty (lower entropy = higher penalty)
            return max(0.0, 1.0 - normalized_entropy)
            
        except Exception:
            return 1.0

class MultiHorizonCompetitionLoss(nn.Module):
    """Competition loss for multi-horizon predictions"""
    
    def __init__(self, horizons: List[str], horizon_weights: Optional[Dict[str, float]] = None,
                 normality_weight: float = 0.7, accuracy_weight: float = 0.3):
        super().__init__()
        self.horizons = horizons
        self.normality_weight = normality_weight
        self.accuracy_weight = accuracy_weight
        
        # Default horizon weights (emphasize shorter horizons)
        if horizon_weights is None:
            self.horizon_weights = {
                '15min': 0.30,
                '30min': 0.25,
                '1hour': 0.20,
                '2hour': 0.15,
                '24hour': 0.10
            }
        else:
            self.horizon_weights = horizon_weights
        
        # Normalize weights
        total_weight = sum(self.horizon_weights.values())
        self.horizon_weights = {k: v/total_weight for k, v in self.horizon_weights.items()}
        
        # Individual loss functions for each horizon
        self.horizon_losses = nn.ModuleDict({
            horizon: CompetitionLoss(normality_weight, accuracy_weight)
            for horizon in horizons
        })
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute multi-horizon competition loss
        
        Args:
            predictions: Dict of {horizon: prediction_tensor}
            targets: Dict of {horizon: target_tensor}
            
        Returns:
            loss: Weighted combination of all horizon losses
            metrics: Detailed breakdown by horizon
        """
        total_loss = 0.0
        all_metrics = {'horizon_losses': {}, 'horizon_weights': self.horizon_weights}
        
        for horizon in self.horizons:
            if horizon in predictions and horizon in targets:
                horizon_loss, horizon_metrics = self.horizon_losses[horizon](
                    predictions[horizon], targets[horizon]
                )
                
                # Weight by horizon importance
                weighted_loss = self.horizon_weights.get(horizon, 0.0) * horizon_loss
                total_loss += weighted_loss
                
                # Store metrics
                all_metrics['horizon_losses'][horizon] = {
                    'loss': horizon_loss.item(),
                    'weighted_loss': weighted_loss.item(),
                    'weight': self.horizon_weights.get(horizon, 0.0),
                    **horizon_metrics
                }
        
        all_metrics['total_loss'] = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
        
        return total_loss, all_metrics

class DualPredictorCompetitionLoss(nn.Module):
    """Competition loss for dual clock/ephemeris predictors"""
    
    def __init__(self, horizons: List[str], 
                 clock_weight: float = 0.4, ephemeris_weight: float = 0.6,
                 normality_weight: float = 0.7, accuracy_weight: float = 0.3):
        super().__init__()
        self.horizons = horizons
        self.clock_weight = clock_weight
        self.ephemeris_weight = ephemeris_weight
        
        # Normalize weights
        total_weight = clock_weight + ephemeris_weight
        self.clock_weight = clock_weight / total_weight
        self.ephemeris_weight = ephemeris_weight / total_weight
        
        # Loss functions for each prediction type
        self.clock_loss = MultiHorizonCompetitionLoss(horizons, normality_weight=normality_weight, accuracy_weight=accuracy_weight)
        self.ephemeris_loss = MultiHorizonCompetitionLoss(horizons, normality_weight=normality_weight, accuracy_weight=accuracy_weight)
    
    def forward(self, predictions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]], 
                targets: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute loss for dual predictor
        
        Args:
            predictions: {horizon: {'clock': (pred, unc), 'ephemeris': (pred, unc)}}
            targets: {horizon: {'clock': target, 'ephemeris': target}}
        """
        # Separate clock and ephemeris predictions
        clock_predictions = {}
        ephemeris_predictions = {}
        clock_targets = {}
        ephemeris_targets = {}
        
        for horizon in self.horizons:
            if horizon in predictions and horizon in targets:
                # Extract predictions (ignore uncertainties for loss computation)
                clock_pred, _ = predictions[horizon]['clock']
                ephemeris_pred, _ = predictions[horizon]['ephemeris']
                
                clock_predictions[horizon] = clock_pred
                ephemeris_predictions[horizon] = ephemeris_pred
                clock_targets[horizon] = targets[horizon]['clock']
                ephemeris_targets[horizon] = targets[horizon]['ephemeris']
        
        # Compute separate losses
        clock_loss, clock_metrics = self.clock_loss(clock_predictions, clock_targets)
        ephemeris_loss, ephemeris_metrics = self.ephemeris_loss(ephemeris_predictions, ephemeris_targets)
        
        # Combined loss
        total_loss = (self.clock_weight * clock_loss + 
                     self.ephemeris_weight * ephemeris_loss)
        
        # Combined metrics
        metrics = {
            'total_loss': total_loss.item(),
            'clock_loss': clock_loss.item(),
            'ephemeris_loss': ephemeris_loss.item(),
            'clock_weight': self.clock_weight,
            'ephemeris_weight': self.ephemeris_weight,
            'clock_metrics': clock_metrics,
            'ephemeris_metrics': ephemeris_metrics
        }
        
        return total_loss, metrics

# Factory functions
def create_competition_loss(normality_weight: float = 0.7, accuracy_weight: float = 0.3) -> CompetitionLoss:
    """Create standard competition loss function"""
    return CompetitionLoss(normality_weight, accuracy_weight)

def create_multi_horizon_competition_loss(horizons: List[str], **kwargs) -> MultiHorizonCompetitionLoss:
    """Create multi-horizon competition loss function"""
    return MultiHorizonCompetitionLoss(horizons, **kwargs)

def create_dual_predictor_competition_loss(horizons: List[str], **kwargs) -> DualPredictorCompetitionLoss:
    """Create dual predictor competition loss function"""
    return DualPredictorCompetitionLoss(horizons, **kwargs)

# Example usage
if __name__ == "__main__":
    # Test competition loss
    loss_fn = create_competition_loss()
    
    # Dummy data
    predictions = torch.randn(100, 1, requires_grad=True)
    targets = torch.randn(100, 1)
    
    loss, metrics = loss_fn(predictions, targets)
    print("Competition Loss Test:")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Test multi-horizon loss
    horizons = ['15min', '30min', '1hour', '2hour', '24hour']
    multi_loss_fn = create_multi_horizon_competition_loss(horizons)
    
    pred_dict = {h: torch.randn(50, 1, requires_grad=True) for h in horizons}
    target_dict = {h: torch.randn(50, 1) for h in horizons}
    
    multi_loss, multi_metrics = multi_loss_fn(pred_dict, target_dict)
    print(f"\nMulti-Horizon Loss: {multi_loss.item():.4f}")
    print(f"Horizon breakdown: {multi_metrics['horizon_losses']}")
