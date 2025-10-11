"""
Competition-Specific Evaluation Framework for GNSS Error Prediction
Implements all required metrics, normality tests, and validation procedures
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

@dataclass
class CompetitionMetrics:
    """Container for all competition evaluation metrics"""
    rmse_15min: float
    rmse_30min: float
    rmse_1hour: float
    rmse_2hour: float
    rmse_24hour: float
    mae_15min: float
    mae_30min: float
    mae_1hour: float
    mae_2hour: float
    mae_24hour: float
    normality_ks_stat: float
    normality_ks_pvalue: float
    normality_sw_stat: float
    normality_sw_pvalue: float
    normality_ad_stat: float
    normality_jb_stat: float
    normality_jb_pvalue: float
    residual_mean: float
    residual_std: float
    residual_skewness: float
    residual_kurtosis: float
    overall_score: float

class NormalityTester:
    """Comprehensive normality testing suite"""
    
    @staticmethod
    def kolmogorov_smirnov_test(residuals: np.ndarray) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test for normality"""
        # Standardize residuals
        standardized = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8)
        ks_stat, p_value = stats.kstest(standardized, 'norm')
        return ks_stat, p_value
    
    @staticmethod
    def shapiro_wilk_test(residuals: np.ndarray) -> Tuple[float, float]:
        """Shapiro-Wilk test for normality (for samples <= 5000)"""
        if len(residuals) > 5000:
            # Sample for large datasets
            sample_indices = np.random.choice(len(residuals), 5000, replace=False)
            residuals = residuals[sample_indices]
        
        standardized = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8)
        sw_stat, p_value = stats.shapiro(standardized)
        return sw_stat, p_value
    
    @staticmethod
    def anderson_darling_test(residuals: np.ndarray) -> float:
        """Anderson-Darling test for normality"""
        standardized = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8)
        ad_stat, _, _ = stats.anderson(standardized, dist='norm')
        return ad_stat
    
    @staticmethod
    def jarque_bera_test(residuals: np.ndarray) -> Tuple[float, float]:
        """Jarque-Bera test for normality"""
        jb_stat, p_value = stats.jarque_bera(residuals)
        return jb_stat, p_value
    
    @staticmethod
    def comprehensive_normality_assessment(residuals: np.ndarray) -> Dict[str, Any]:
        """Run all normality tests and return comprehensive results"""
        results = {}
        
        # Basic statistics
        results['mean'] = np.mean(residuals)
        results['std'] = np.std(residuals)
        results['skewness'] = stats.skew(residuals)
        results['kurtosis'] = stats.kurtosis(residuals)
        
        # Normality tests
        try:
            results['ks_stat'], results['ks_pvalue'] = NormalityTester.kolmogorov_smirnov_test(residuals)
        except Exception as e:
            results['ks_stat'], results['ks_pvalue'] = np.nan, np.nan
        
        try:
            results['sw_stat'], results['sw_pvalue'] = NormalityTester.shapiro_wilk_test(residuals)
        except Exception as e:
            results['sw_stat'], results['sw_pvalue'] = np.nan, np.nan
        
        try:
            results['ad_stat'] = NormalityTester.anderson_darling_test(residuals)
        except Exception as e:
            results['ad_stat'] = np.nan
        
        try:
            results['jb_stat'], results['jb_pvalue'] = NormalityTester.jarque_bera_test(residuals)
        except Exception as e:
            results['jb_stat'], results['jb_pvalue'] = np.nan, np.nan
        
        # Composite normality score (lower is better)
        normality_scores = []
        if not np.isnan(results['ks_stat']):
            normality_scores.append(results['ks_stat'])
        if not np.isnan(results['sw_stat']):
            normality_scores.append(1.0 - results['sw_stat'])  # Convert to penalty
        if not np.isnan(results['ad_stat']):
            normality_scores.append(results['ad_stat'] / 10.0)  # Scale down
        
        normality_scores.extend([abs(results['skewness']), abs(results['kurtosis']) / 3.0])
        
        results['composite_normality_score'] = np.mean(normality_scores) if normality_scores else 1.0
        
        return results

class CompetitionEvaluator:
    """Main evaluation class for competition metrics"""
    
    def __init__(self, horizons: List[str] = ['15min', '30min', '1hour', '2hour', '24hour']):
        self.horizons = horizons
        self.horizon_weights = {
            '15min': 0.25,
            '30min': 0.25,
            '1hour': 0.2,
            '2hour': 0.15,
            '24hour': 0.15
        }
    
    def evaluate_predictions(self, predictions: Dict[str, np.ndarray], 
                           targets: Dict[str, np.ndarray]) -> CompetitionMetrics:
        """Comprehensive evaluation of predictions"""
        
        # Initialize metrics dictionary
        metrics_dict = {}
        all_residuals = []
        
        # Evaluate each horizon
        for horizon in self.horizons:
            if horizon in predictions and horizon in targets:
                pred = predictions[horizon].flatten()
                target = targets[horizon].flatten()
                
                # Basic metrics
                rmse = np.sqrt(mean_squared_error(target, pred))
                mae = mean_absolute_error(target, pred)
                
                metrics_dict[f'rmse_{horizon}'] = rmse
                metrics_dict[f'mae_{horizon}'] = mae
                
                # Collect residuals for normality testing
                residuals = pred - target
                all_residuals.extend(residuals.tolist())
        
        # Comprehensive normality assessment
        if all_residuals:
            all_residuals = np.array(all_residuals)
            normality_results = NormalityTester.comprehensive_normality_assessment(all_residuals)
            
            # Add normality metrics
            metrics_dict.update({
                'normality_ks_stat': normality_results.get('ks_stat', np.nan),
                'normality_ks_pvalue': normality_results.get('ks_pvalue', np.nan),
                'normality_sw_stat': normality_results.get('sw_stat', np.nan),
                'normality_sw_pvalue': normality_results.get('sw_pvalue', np.nan),
                'normality_ad_stat': normality_results.get('ad_stat', np.nan),
                'normality_jb_stat': normality_results.get('jb_stat', np.nan),
                'normality_jb_pvalue': normality_results.get('jb_pvalue', np.nan),
                'residual_mean': normality_results['mean'],
                'residual_std': normality_results['std'],
                'residual_skewness': normality_results['skewness'],
                'residual_kurtosis': normality_results['kurtosis']
            })
            
            # Calculate overall competition score
            overall_score = self._calculate_overall_score(metrics_dict, normality_results)
            metrics_dict['overall_score'] = overall_score
        
        else:
            # Fill with NaN if no valid predictions
            for key in ['normality_ks_stat', 'normality_ks_pvalue', 'normality_sw_stat', 
                       'normality_sw_pvalue', 'normality_ad_stat', 'normality_jb_stat',
                       'normality_jb_pvalue', 'residual_mean', 'residual_std',
                       'residual_skewness', 'residual_kurtosis', 'overall_score']:
                metrics_dict[key] = np.nan
        
        # Create CompetitionMetrics object
        return CompetitionMetrics(**metrics_dict)
    
    def _calculate_overall_score(self, metrics_dict: Dict[str, float], 
                               normality_results: Dict[str, Any]) -> float:
        """Calculate overall competition score (lower is better)"""
        
        # Weighted RMSE across horizons
        rmse_score = 0.0
        for horizon in self.horizons:
            rmse_key = f'rmse_{horizon}'
            if rmse_key in metrics_dict and not np.isnan(metrics_dict[rmse_key]):
                rmse_score += self.horizon_weights[horizon] * metrics_dict[rmse_key]
        
        # Normality penalty (scaled to be comparable with RMSE)
        normality_penalty = normality_results.get('composite_normality_score', 1.0)
        
        # Combined score (70% accuracy, 30% normality)
        overall_score = 0.7 * rmse_score + 0.3 * normality_penalty
        
        return overall_score
    
    def cross_validate_day8_prediction(self, model, data: pd.DataFrame, 
                                     n_folds: int = 5) -> Dict[str, Any]:
        """Cross-validation specifically for day-8 prediction scenario"""
        
        # Group data by days
        data['day'] = data['timestamp'].dt.day
        unique_days = sorted(data['day'].unique())
        
        if len(unique_days) < 8:
            raise ValueError("Need at least 8 days of data for day-8 prediction validation")
        
        cv_results = {horizon: {'rmse': [], 'mae': [], 'normality': []} for horizon in self.horizons}
        
        # Simulate day-8 prediction scenario
        for fold in range(n_folds):
            # Use first 7 days for training, 8th day for testing
            if fold + 8 <= len(unique_days):
                train_days = unique_days[fold:fold+7]
                test_day = unique_days[fold+7]
                
                train_data = data[data['day'].isin(train_days)]
                test_data = data[data['day'] == test_day]
                
                # Train model (placeholder - implement actual training)
                # model.fit(train_data)
                
                # Make predictions for test day
                # predictions = model.predict(test_data)
                
                # For now, simulate predictions
                predictions = self._simulate_predictions(test_data)
                targets = self._extract_targets(test_data)
                
                # Evaluate this fold
                fold_metrics = self.evaluate_predictions(predictions, targets)
                
                # Store results
                for horizon in self.horizons:
                    rmse_key = f'rmse_{horizon}'
                    mae_key = f'mae_{horizon}'
                    
                    if hasattr(fold_metrics, rmse_key):
                        cv_results[horizon]['rmse'].append(getattr(fold_metrics, rmse_key))
                        cv_results[horizon]['mae'].append(getattr(fold_metrics, mae_key))
        
        # Aggregate cross-validation results
        aggregated_results = {}
        for horizon in self.horizons:
            if cv_results[horizon]['rmse']:
                aggregated_results[f'{horizon}_rmse_mean'] = np.mean(cv_results[horizon]['rmse'])
                aggregated_results[f'{horizon}_rmse_std'] = np.std(cv_results[horizon]['rmse'])
                aggregated_results[f'{horizon}_mae_mean'] = np.mean(cv_results[horizon]['mae'])
                aggregated_results[f'{horizon}_mae_std'] = np.std(cv_results[horizon]['mae'])
        
        return aggregated_results
    
    def _simulate_predictions(self, test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Simulate predictions for testing (replace with actual model predictions)"""
        n_samples = len(test_data)
        predictions = {}
        
        for horizon in self.horizons:
            # Simulate predictions with some noise
            predictions[horizon] = np.random.normal(0, 0.1, n_samples)
        
        return predictions
    
    def _extract_targets(self, test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract target values for different horizons"""
        targets = {}
        
        for horizon in self.horizons:
            # For simulation, use error column as target
            targets[horizon] = test_data['error'].values
        
        return targets

class VisualizationGenerator:
    """Generate competition-required visualizations"""
    
    @staticmethod
    def create_qq_plot(residuals: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
        """Create Q-Q plot for normality assessment"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Standardize residuals
        standardized = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8)
        
        # Create Q-Q plot
        stats.probplot(standardized, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot: Residuals vs Normal Distribution", fontsize=14)
        ax.set_xlabel("Theoretical Quantiles", fontsize=12)
        ax.set_ylabel("Sample Quantiles", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add R² value
        slope, intercept, r_value = stats.probplot(standardized, dist="norm")[:2][:3]
        ax.text(0.05, 0.95, f'R² = {r_value**2:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_residual_distribution_plot(residuals: np.ndarray, 
                                        save_path: Optional[str] = None) -> plt.Figure:
        """Create residual distribution plot with normality overlay"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Standardize residuals
        standardized = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8)
        
        # Histogram with normal overlay
        ax1.hist(standardized, bins=50, density=True, alpha=0.7, color='skyblue', 
                edgecolor='black', label='Residuals')
        
        # Overlay normal distribution
        x = np.linspace(standardized.min(), standardized.max(), 100)
        normal_pdf = stats.norm.pdf(x, 0, 1)
        ax1.plot(x, normal_pdf, 'r-', linewidth=2, label='Normal Distribution')
        
        ax1.set_xlabel('Standardized Residuals')
        ax1.set_ylabel('Density')
        ax1.set_title('Residual Distribution vs Normal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(standardized, vert=True)
        ax2.set_ylabel('Standardized Residuals')
        ax2.set_title('Residual Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_horizon_performance_plot(metrics: CompetitionMetrics, 
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Create performance plot across different horizons"""
        horizons = ['15min', '30min', '1hour', '2hour', '24hour']
        
        rmse_values = [
            metrics.rmse_15min, metrics.rmse_30min, metrics.rmse_1hour,
            metrics.rmse_2hour, metrics.rmse_24hour
        ]
        
        mae_values = [
            metrics.mae_15min, metrics.mae_30min, metrics.mae_1hour,
            metrics.mae_2hour, metrics.mae_24hour
        ]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE plot
        ax1.bar(horizons, rmse_values, color='lightcoral', alpha=0.7)
        ax1.set_xlabel('Prediction Horizon')
        ax1.set_ylabel('RMSE')
        ax1.set_title('RMSE Across Prediction Horizons')
        ax1.tick_params(axis='x', rotation=45)
        
        # MAE plot
        ax2.bar(horizons, mae_values, color='lightblue', alpha=0.7)
        ax2.set_xlabel('Prediction Horizon')
        ax2.set_ylabel('MAE')
        ax2.set_title('MAE Across Prediction Horizons')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def run_competition_evaluation(predictions: Dict[str, np.ndarray], 
                             targets: Dict[str, np.ndarray],
                             output_dir: str = "competition_results") -> CompetitionMetrics:
    """Run complete competition evaluation and generate all required outputs"""
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = CompetitionEvaluator()
    
    # Run evaluation
    metrics = evaluator.evaluate_predictions(predictions, targets)
    
    # Generate visualizations
    all_residuals = []
    for horizon in evaluator.horizons:
        if horizon in predictions and horizon in targets:
            residuals = predictions[horizon].flatten() - targets[horizon].flatten()
            all_residuals.extend(residuals.tolist())
    
    if all_residuals:
        all_residuals = np.array(all_residuals)
        
        # Q-Q plot
        VisualizationGenerator.create_qq_plot(
            all_residuals, 
            save_path=os.path.join(output_dir, "qq_plot.png")
        )
        
        # Distribution plot
        VisualizationGenerator.create_residual_distribution_plot(
            all_residuals,
            save_path=os.path.join(output_dir, "residual_distribution.png")
        )
        
        # Performance plot
        VisualizationGenerator.create_horizon_performance_plot(
            metrics,
            save_path=os.path.join(output_dir, "horizon_performance.png")
        )
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics.__dict__])
    metrics_df.to_csv(os.path.join(output_dir, "competition_metrics.csv"), index=False)
    
    return metrics
