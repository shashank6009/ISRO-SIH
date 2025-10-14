"""
Enhanced Competition Evaluation System for GNSS Error Prediction
Implements competition-specific scoring with normality as primary criterion (70%) and accuracy as secondary (30%)
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

@dataclass
class CompetitionSubmissionMetrics:
    """Complete metrics for competition submission evaluation"""
    # Primary Score (70% weight)
    normality_score: float
    normality_breakdown: Dict[str, float]
    
    # Secondary Score (30% weight)
    accuracy_score: float
    accuracy_breakdown: Dict[str, float]
    
    # Final Competition Score
    final_score: float
    
    # Detailed horizon-wise metrics
    horizon_metrics: Dict[str, Dict[str, float]]
    
    # Orbit-specific metrics
    orbit_metrics: Dict[str, Dict[str, float]]
    
    # Model comparison
    model_ranking: List[Dict[str, Any]]
    
    # Statistical significance tests
    statistical_tests: Dict[str, Dict[str, float]]

class AdvancedNormalityTester:
    """Advanced normality testing with multiple statistical methods"""
    
    @staticmethod
    def comprehensive_normality_assessment(residuals: np.ndarray, 
                                         confidence_level: float = 0.05) -> Dict[str, Any]:
        """
        Perform comprehensive normality assessment using multiple tests
        Returns normalized scores where higher = better normality
        """
        if len(residuals) < 8:
            return {'error': 'insufficient_samples', 'normality_score': 0.0}
        
        # Standardize residuals
        standardized = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-8)
        
        results = {}
        normality_scores = []
        
        # 1. Kolmogorov-Smirnov Test
        try:
            ks_stat, ks_p = stats.kstest(standardized, 'norm')
            # Convert to normality score (higher p-value = better normality)
            ks_score = min(ks_p / confidence_level, 1.0)  # Normalize by confidence level
            normality_scores.append(ks_score)
            results['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'normality_score': ks_score,
                'passes_test': ks_p > confidence_level
            }
        except Exception as e:
            results['kolmogorov_smirnov'] = {'error': str(e), 'normality_score': 0.0}
        
        # 2. Anderson-Darling Test
        try:
            ad_result = stats.anderson(standardized, dist='norm')
            # Anderson-Darling: smaller statistic = better normality
            # Normalize using critical values
            critical_5pct = ad_result.critical_values[2]  # 5% significance level
            ad_score = max(0.0, 1.0 - (ad_result.statistic / critical_5pct))
            normality_scores.append(ad_score)
            results['anderson_darling'] = {
                'statistic': ad_result.statistic,
                'critical_values': ad_result.critical_values.tolist(),
                'significance_levels': ad_result.significance_level.tolist(),
                'normality_score': ad_score,
                'passes_test': ad_result.statistic < critical_5pct
            }
        except Exception as e:
            results['anderson_darling'] = {'error': str(e), 'normality_score': 0.0}
        
        # 3. Shapiro-Wilk Test (for smaller samples)
        if len(standardized) <= 5000:
            try:
                sw_stat, sw_p = stats.shapiro(standardized[:5000])
                sw_score = min(sw_p / confidence_level, 1.0)
                normality_scores.append(sw_score)
                results['shapiro_wilk'] = {
                    'statistic': sw_stat,
                    'p_value': sw_p,
                    'normality_score': sw_score,
                    'passes_test': sw_p > confidence_level
                }
            except Exception as e:
                results['shapiro_wilk'] = {'error': str(e), 'normality_score': 0.0}
        
        # 4. Jarque-Bera Test
        try:
            jb_stat, jb_p = stats.jarque_bera(standardized)
            jb_score = min(jb_p / confidence_level, 1.0)
            normality_scores.append(jb_score)
            results['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'normality_score': jb_score,
                'passes_test': jb_p > confidence_level
            }
        except Exception as e:
            results['jarque_bera'] = {'error': str(e), 'normality_score': 0.0}
        
        # 5. D'Agostino-Pearson Test
        try:
            dp_stat, dp_p = stats.normaltest(standardized)
            dp_score = min(dp_p / confidence_level, 1.0)
            normality_scores.append(dp_score)
            results['dagostino_pearson'] = {
                'statistic': dp_stat,
                'p_value': dp_p,
                'normality_score': dp_score,
                'passes_test': dp_p > confidence_level
            }
        except Exception as e:
            results['dagostino_pearson'] = {'error': str(e), 'normality_score': 0.0}
        
        # 6. Histogram-based normality score
        try:
            hist_score = AdvancedNormalityTester._compute_histogram_normality(standardized)
            normality_scores.append(hist_score)
            results['histogram_normality'] = {
                'normality_score': hist_score
            }
        except Exception as e:
            results['histogram_normality'] = {'error': str(e), 'normality_score': 0.0}
        
        # Composite normality score
        if normality_scores:
            composite_score = np.mean(normality_scores)
            # Apply additional penalty for extreme outliers
            outlier_penalty = AdvancedNormalityTester._compute_outlier_penalty(standardized)
            composite_score *= (1.0 - outlier_penalty)
        else:
            composite_score = 0.0
        
        results['composite_normality_score'] = max(0.0, min(1.0, composite_score))
        results['num_tests_passed'] = sum(1 for test_name, test_result in results.items() 
                                        if isinstance(test_result, dict) and 
                                        test_result.get('passes_test', False))
        results['total_tests'] = len([k for k in results.keys() if k not in ['composite_normality_score', 'num_tests_passed', 'total_tests']])
        
        return results
    
    @staticmethod
    def _compute_histogram_normality(data: np.ndarray) -> float:
        """Compute normality based on histogram similarity to normal distribution"""
        try:
            # Create histogram
            hist, bin_edges = np.histogram(data, bins=min(50, len(data) // 20), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Expected normal distribution
            expected_hist = stats.norm.pdf(bin_centers, loc=0, scale=1)
            
            # Compute similarity (using negative KL divergence, normalized)
            # Add small epsilon to avoid log(0)
            hist_smooth = hist + 1e-10
            expected_smooth = expected_hist + 1e-10
            
            # Normalize histograms
            hist_norm = hist_smooth / np.sum(hist_smooth)
            expected_norm = expected_smooth / np.sum(expected_smooth)
            
            # Compute KL divergence
            kl_div = np.sum(hist_norm * np.log(hist_norm / expected_norm))
            
            # Convert to similarity score (0 = identical, higher = more different)
            # Use exponential decay to convert to 0-1 score
            similarity_score = np.exp(-kl_div)
            
            return min(1.0, max(0.0, similarity_score))
            
        except Exception:
            return 0.5  # Neutral score if computation fails
    
    @staticmethod
    def _compute_outlier_penalty(data: np.ndarray) -> float:
        """Compute penalty for extreme outliers"""
        try:
            # Count outliers beyond 3 standard deviations
            outliers = np.abs(data) > 3.0
            outlier_ratio = np.sum(outliers) / len(data)
            
            # Expected outlier ratio for normal distribution (~0.27%)
            expected_outlier_ratio = 0.0027
            
            # Penalty increases with excess outliers
            excess_outliers = max(0, outlier_ratio - expected_outlier_ratio)
            penalty = min(0.5, excess_outliers * 10)  # Cap penalty at 50%
            
            return penalty
            
        except Exception:
            return 0.0

class CompetitionEvaluator:
    """Enhanced competition evaluator with normality as primary criterion"""
    
    def __init__(self, horizons: List[str] = ['15min', '30min', '1hour', '2hour', '24hour']):
        self.horizons = horizons
        
        # Competition weights: Normality (70%) + Accuracy (30%)
        self.evaluation_weights = {
            'normality_score': 0.70,    # PRIMARY: Normal distribution closeness
            'accuracy_score': 0.30      # SECONDARY: Prediction accuracy
        }
        
        # Horizon weights (emphasize shorter horizons)
        self.horizon_weights = {
            '15min': 0.30,
            '30min': 0.25,
            '1hour': 0.20,
            '2hour': 0.15,
            '24hour': 0.10
        }
        
        # Normalize horizon weights
        total_weight = sum(self.horizon_weights.values())
        self.horizon_weights = {k: v/total_weight for k, v in self.horizon_weights.items()}
    
    def evaluate_competition_submission(self, 
                                      predictions: Dict[str, Dict[str, np.ndarray]], 
                                      targets: Dict[str, Dict[str, np.ndarray]],
                                      orbit_classes: Optional[List[str]] = None,
                                      model_names: Optional[List[str]] = None) -> CompetitionSubmissionMetrics:
        """
        Comprehensive evaluation of competition submission
        
        Args:
            predictions: {model_name: {horizon: predictions}}
            targets: {horizon: targets}
            orbit_classes: List of orbit classes for orbit-specific evaluation
            model_names: List of model names for ranking
        """
        
        if model_names is None:
            model_names = list(predictions.keys())
        
        all_model_results = {}
        
        # Evaluate each model
        for model_name in model_names:
            if model_name in predictions:
                model_results = self._evaluate_single_model(
                    predictions[model_name], targets, orbit_classes
                )
                all_model_results[model_name] = model_results
        
        # Rank models by final score
        model_ranking = self._rank_models(all_model_results)
        
        # Get best model results
        best_model = model_ranking[0] if model_ranking else None
        best_results = all_model_results.get(best_model['model_name'], {}) if best_model else {}
        
        # Create comprehensive metrics
        submission_metrics = CompetitionSubmissionMetrics(
            normality_score=best_results.get('normality_score', 0.0),
            normality_breakdown=best_results.get('normality_breakdown', {}),
            accuracy_score=best_results.get('accuracy_score', 0.0),
            accuracy_breakdown=best_results.get('accuracy_breakdown', {}),
            final_score=best_results.get('final_score', 0.0),
            horizon_metrics=best_results.get('horizon_metrics', {}),
            orbit_metrics=best_results.get('orbit_metrics', {}),
            model_ranking=model_ranking,
            statistical_tests=best_results.get('statistical_tests', {})
        )
        
        return submission_metrics
    
    def _evaluate_single_model(self, 
                              model_predictions: Dict[str, np.ndarray],
                              targets: Dict[str, np.ndarray],
                              orbit_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate a single model's predictions"""
        
        horizon_metrics = {}
        all_residuals = []
        weighted_accuracy_scores = []
        weighted_normality_scores = []
        
        # Evaluate each horizon
        for horizon in self.horizons:
            if horizon in model_predictions and horizon in targets:
                pred = model_predictions[horizon].flatten()
                target = targets[horizon].flatten()
                
                # Accuracy metrics
                rmse = np.sqrt(mean_squared_error(target, pred))
                mae = mean_absolute_error(target, pred)
                
                # Residuals for normality testing
                residuals = pred - target
                all_residuals.extend(residuals.tolist())
                
                # Normality assessment
                normality_results = AdvancedNormalityTester.comprehensive_normality_assessment(residuals)
                normality_score = normality_results.get('composite_normality_score', 0.0)
                
                # Accuracy score (normalized RMSE, inverted so higher is better)
                # Use robust normalization based on target standard deviation
                target_std = np.std(target) + 1e-8
                accuracy_score = max(0.0, 1.0 - (rmse / target_std))
                
                # Store horizon metrics
                horizon_metrics[horizon] = {
                    'rmse': rmse,
                    'mae': mae,
                    'accuracy_score': accuracy_score,
                    'normality_score': normality_score,
                    'normality_details': normality_results,
                    'residual_stats': {
                        'mean': np.mean(residuals),
                        'std': np.std(residuals),
                        'skewness': stats.skew(residuals),
                        'kurtosis': stats.kurtosis(residuals)
                    }
                }
                
                # Weighted contributions
                horizon_weight = self.horizon_weights.get(horizon, 0.0)
                weighted_accuracy_scores.append(accuracy_score * horizon_weight)
                weighted_normality_scores.append(normality_score * horizon_weight)
        
        # Overall scores
        overall_accuracy = sum(weighted_accuracy_scores) if weighted_accuracy_scores else 0.0
        overall_normality = sum(weighted_normality_scores) if weighted_normality_scores else 0.0
        
        # Final competition score
        final_score = (self.evaluation_weights['accuracy_score'] * overall_accuracy + 
                      self.evaluation_weights['normality_score'] * overall_normality)
        
        # Overall normality assessment on all residuals
        if all_residuals:
            overall_normality_results = AdvancedNormalityTester.comprehensive_normality_assessment(
                np.array(all_residuals)
            )
        else:
            overall_normality_results = {'composite_normality_score': 0.0}
        
        return {
            'final_score': final_score,
            'accuracy_score': overall_accuracy,
            'normality_score': overall_normality,
            'accuracy_breakdown': {h: m['accuracy_score'] for h, m in horizon_metrics.items()},
            'normality_breakdown': {h: m['normality_score'] for h, m in horizon_metrics.items()},
            'horizon_metrics': horizon_metrics,
            'statistical_tests': overall_normality_results,
            'evaluation_weights': self.evaluation_weights,
            'horizon_weights': self.horizon_weights
        }
    
    def _rank_models(self, model_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank models by competition score"""
        
        rankings = []
        
        for model_name, results in model_results.items():
            rankings.append({
                'model_name': model_name,
                'final_score': results.get('final_score', 0.0),
                'normality_score': results.get('normality_score', 0.0),
                'accuracy_score': results.get('accuracy_score', 0.0),
                'rank': 0  # Will be set after sorting
            })
        
        # Sort by final score (descending)
        rankings.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def generate_competition_report(self, metrics: CompetitionSubmissionMetrics) -> str:
        """Generate detailed competition evaluation report"""
        
        report = []
        report.append("=" * 80)
        report.append("GNSS ERROR PREDICTION COMPETITION EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall scores
        report.append("OVERALL COMPETITION SCORES:")
        report.append(f"Final Score: {metrics.final_score:.4f}")
        report.append(f"Normality Score (70%): {metrics.normality_score:.4f}")
        report.append(f"Accuracy Score (30%): {metrics.accuracy_score:.4f}")
        report.append("")
        
        # Model ranking
        report.append("MODEL RANKING:")
        for i, model in enumerate(metrics.model_ranking):
            rank_suffix = {1: "st", 2: "nd", 3: "rd"}.get(model['rank'], "th")
            report.append(f"{model['rank']}{rank_suffix}: {model['model_name']} - Score: {model['final_score']:.4f}")
        report.append("")
        
        # Horizon breakdown
        report.append("HORIZON-WISE PERFORMANCE:")
        for horizon, metrics_dict in metrics.horizon_metrics.items():
            report.append(f"{horizon}:")
            report.append(f"  RMSE: {metrics_dict['rmse']:.6f}")
            report.append(f"  Normality Score: {metrics_dict['normality_score']:.4f}")
            report.append(f"  Accuracy Score: {metrics_dict['accuracy_score']:.4f}")
        report.append("")
        
        # Statistical tests summary
        if metrics.statistical_tests:
            report.append("NORMALITY STATISTICAL TESTS:")
            tests_passed = metrics.statistical_tests.get('num_tests_passed', 0)
            total_tests = metrics.statistical_tests.get('total_tests', 0)
            report.append(f"Tests Passed: {tests_passed}/{total_tests}")
            
            for test_name, test_result in metrics.statistical_tests.items():
                if isinstance(test_result, dict) and 'p_value' in test_result:
                    status = "PASS" if test_result.get('passes_test', False) else "FAIL"
                    report.append(f"  {test_name}: {status} (p={test_result['p_value']:.4f})")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

# Factory functions
def create_competition_evaluator(horizons: Optional[List[str]] = None) -> CompetitionEvaluator:
    """Create competition evaluator with standard settings"""
    if horizons is None:
        horizons = ['15min', '30min', '1hour', '2hour', '24hour']
    return CompetitionEvaluator(horizons)

def evaluate_competition_models(predictions: Dict[str, Dict[str, np.ndarray]], 
                              targets: Dict[str, np.ndarray],
                              model_names: Optional[List[str]] = None) -> CompetitionSubmissionMetrics:
    """Convenience function for evaluating competition models"""
    evaluator = create_competition_evaluator()
    return evaluator.evaluate_competition_submission(predictions, targets, model_names=model_names)

# Example usage
if __name__ == "__main__":
    # Test the competition evaluator
    horizons = ['15min', '30min', '1hour', '2hour', '24hour']
    
    # Create dummy data
    np.random.seed(42)
    targets = {h: np.random.normal(0, 1, 1000) for h in horizons}
    
    # Create predictions for multiple models
    predictions = {
        'Model_A': {h: targets[h] + np.random.normal(0, 0.1, 1000) for h in horizons},
        'Model_B': {h: targets[h] + np.random.normal(0, 0.2, 1000) for h in horizons},
        'Model_C': {h: targets[h] + np.random.exponential(0.1, 1000) for h in horizons}  # Non-normal errors
    }
    
    # Evaluate
    evaluator = create_competition_evaluator()
    metrics = evaluator.evaluate_competition_submission(predictions, targets)
    
    # Generate report
    report = evaluator.generate_competition_report(metrics)
    print(report)
