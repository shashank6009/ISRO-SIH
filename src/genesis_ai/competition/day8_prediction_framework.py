"""
Day-8 Prediction Framework for GNSS Competition
Specialized framework for predicting errors on the 8th day using 7 days of training data
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from genesis_ai.db.competition_models import CompetitionDataPoint, CompetitionDataLoader
from genesis_ai.models.competition_dual_predictor import CompetitionDualPredictor, DualPredictorConfig
from genesis_ai.training.competition_loss import DualPredictorCompetitionLoss
from genesis_ai.evaluation.enhanced_competition_metrics import CompetitionEvaluator, CompetitionSubmissionMetrics

@dataclass
class Day8PredictionConfig:
    """Configuration for Day-8 prediction framework"""
    # Time windows
    training_days: int = 7
    prediction_intervals_minutes: int = 15
    prediction_horizons: List[str] = None
    
    # Model configuration
    model_config: DualPredictorConfig = None
    
    # Cross-validation settings
    use_cross_validation: bool = True
    cv_folds: int = 5
    validation_split: float = 0.2
    
    # Data preprocessing
    sequence_length: int = 96  # 24 hours of 15-min intervals
    overlap_ratio: float = 0.5
    
    # Training parameters
    max_epochs: int = 100
    early_stopping_patience: int = 15
    learning_rate: float = 0.0001
    batch_size: int = 32
    
    # Competition-specific
    normalize_by_orbit: bool = True
    separate_clock_ephemeris: bool = True
    
    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = ['15min', '30min', '1hour', '2hour', '24hour']
        
        if self.model_config is None:
            self.model_config = DualPredictorConfig(
                normality_weight=0.7,
                accuracy_weight=0.3,
                horizons=self.prediction_horizons
            )

class Day8DataPreprocessor:
    """Specialized data preprocessor for Day-8 prediction"""
    
    def __init__(self, config: Day8PredictionConfig):
        self.config = config
        self.data_loader = CompetitionDataLoader()
    
    def prepare_training_data(self, data_points: List[CompetitionDataPoint]) -> Dict[str, Any]:
        """
        Prepare 7-day training data for Day-8 prediction
        
        Args:
            data_points: List of competition data points
            
        Returns:
            Dict containing processed training data, validation data, and metadata
        """
        
        # Filter training data (first 7 days)
        training_points = [dp for dp in data_points if dp.is_training]
        
        if not training_points:
            raise ValueError("No training data found. Ensure is_training=True for 7-day data.")
        
        # Convert to DataFrame for easier processing
        training_df = self._data_points_to_dataframe(training_points)
        
        # Separate by orbit class
        orbit_data = {}
        for orbit_class in ['GEO', 'GSO', 'MEO']:
            orbit_df = training_df[training_df['orbit_class'] == orbit_class].copy()
            if not orbit_df.empty:
                orbit_data[orbit_class] = self._prepare_orbit_data(orbit_df, orbit_class)
        
        # Create sequences for training
        training_sequences = self._create_training_sequences(orbit_data)
        
        # Split into train/validation
        train_data, val_data = self._split_train_validation(training_sequences)
        
        return {
            'train_data': train_data,
            'val_data': val_data,
            'orbit_data': orbit_data,
            'metadata': {
                'total_points': len(training_points),
                'orbit_distribution': {oc: len(od['data']) for oc, od in orbit_data.items()},
                'sequence_length': self.config.sequence_length,
                'prediction_horizons': self.config.prediction_horizons
            }
        }
    
    def prepare_day8_targets(self, data_points: List[CompetitionDataPoint]) -> Dict[str, Any]:
        """
        Prepare Day-8 target data for prediction
        
        Args:
            data_points: List of competition data points including Day-8 data
            
        Returns:
            Dict containing Day-8 target sequences and metadata
        """
        
        # Filter Day-8 data
        day8_points = [dp for dp in data_points if not dp.is_training]
        
        if not day8_points:
            raise ValueError("No Day-8 data found. Ensure is_training=False for Day-8 data.")
        
        # Convert to DataFrame
        day8_df = self._data_points_to_dataframe(day8_points)
        
        # Separate by orbit class
        day8_orbit_data = {}
        for orbit_class in ['GEO', 'GSO', 'MEO']:
            orbit_df = day8_df[day8_df['orbit_class'] == orbit_class].copy()
            if not orbit_df.empty:
                day8_orbit_data[orbit_class] = self._prepare_orbit_data(orbit_df, orbit_class)
        
        # Create target sequences for each prediction horizon
        target_sequences = self._create_day8_target_sequences(day8_orbit_data)
        
        return {
            'target_data': target_sequences,
            'orbit_data': day8_orbit_data,
            'metadata': {
                'total_points': len(day8_points),
                'orbit_distribution': {oc: len(od['data']) for oc, od in day8_orbit_data.items()},
                'prediction_horizons': self.config.prediction_horizons
            }
        }
    
    def _data_points_to_dataframe(self, data_points: List[CompetitionDataPoint]) -> pd.DataFrame:
        """Convert data points to pandas DataFrame"""
        
        rows = []
        for dp in data_points:
            row = {
                'satellite_id': dp.satellite_id,
                'timestamp': dp.timestamp,
                'orbit_class': dp.orbit_class,
                'clock_error': dp.clock_error,
                'broadcast_clock_bias': dp.broadcast_clock_bias,
                'modeled_clock_bias': dp.modeled_clock_bias,
                'data_quality': dp.data_quality,
                'is_training': dp.is_training
            }
            
            # Add ephemeris data
            if dp.ephemeris_errors:
                for key, value in dp.ephemeris_errors.items():
                    row[f'ephemeris_{key}'] = value
            
            if dp.broadcast_ephemeris:
                for key, value in dp.broadcast_ephemeris.items():
                    row[f'broadcast_{key}'] = value
            
            if dp.modeled_ephemeris:
                for key, value in dp.modeled_ephemeris.items():
                    row[f'modeled_{key}'] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values(['satellite_id', 'timestamp'])
        
        return df
    
    def _prepare_orbit_data(self, orbit_df: pd.DataFrame, orbit_class: str) -> Dict[str, Any]:
        """Prepare data for specific orbit class"""
        
        # Normalize timestamps
        orbit_df = orbit_df.copy()
        orbit_df['timestamp_normalized'] = (
            orbit_df['timestamp'] - orbit_df['timestamp'].min()
        ).dt.total_seconds() / 3600  # Hours since start
        
        # Create feature matrix
        feature_columns = [
            'clock_error', 'broadcast_clock_bias', 'modeled_clock_bias',
            'timestamp_normalized', 'data_quality'
        ]
        
        # Add ephemeris features if available
        ephemeris_columns = [col for col in orbit_df.columns if col.startswith('ephemeris_')]
        feature_columns.extend(ephemeris_columns)
        
        # Add broadcast/modeled ephemeris features
        broadcast_columns = [col for col in orbit_df.columns if col.startswith('broadcast_') and col != 'broadcast_clock_bias']
        modeled_columns = [col for col in orbit_df.columns if col.startswith('modeled_') and col != 'modeled_clock_bias']
        feature_columns.extend(broadcast_columns + modeled_columns)
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in orbit_df.columns]
        
        feature_matrix = orbit_df[available_features].values
        
        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        return {
            'data': orbit_df,
            'features': feature_matrix,
            'feature_names': available_features,
            'satellites': orbit_df['satellite_id'].unique().tolist(),
            'orbit_class': orbit_class
        }
    
    def _create_training_sequences(self, orbit_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Create training sequences from orbit data"""
        
        sequences = {}
        
        for orbit_class, data_dict in orbit_data.items():
            orbit_sequences = []
            
            for satellite_id in data_dict['satellites']:
                sat_data = data_dict['data'][data_dict['data']['satellite_id'] == satellite_id]
                sat_features = sat_data[data_dict['feature_names']].values
                
                # Create overlapping sequences
                seq_len = self.config.sequence_length
                step_size = max(1, int(seq_len * (1 - self.config.overlap_ratio)))
                
                for i in range(0, len(sat_features) - seq_len + 1, step_size):
                    sequence = {
                        'satellite_id': satellite_id,
                        'orbit_class': orbit_class,
                        'features': sat_features[i:i+seq_len],
                        'timestamps': sat_data.iloc[i:i+seq_len]['timestamp'].values,
                        'clock_targets': sat_data.iloc[i:i+seq_len]['clock_error'].values,
                        'ephemeris_targets': self._extract_ephemeris_targets(sat_data.iloc[i:i+seq_len])
                    }
                    orbit_sequences.append(sequence)
            
            sequences[orbit_class] = orbit_sequences
        
        return sequences
    
    def _create_day8_target_sequences(self, day8_orbit_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create Day-8 target sequences for each prediction horizon"""
        
        target_sequences = {}
        
        for orbit_class, data_dict in day8_orbit_data.items():
            orbit_targets = {}
            
            for horizon in self.config.prediction_horizons:
                horizon_minutes = self._horizon_to_minutes(horizon)
                horizon_targets = []
                
                for satellite_id in data_dict['satellites']:
                    sat_data = data_dict['data'][data_dict['data']['satellite_id'] == satellite_id]
                    
                    # Create targets at specified horizon intervals
                    for i in range(0, len(sat_data), horizon_minutes // self.config.prediction_intervals_minutes):
                        if i < len(sat_data):
                            target = {
                                'satellite_id': satellite_id,
                                'orbit_class': orbit_class,
                                'timestamp': sat_data.iloc[i]['timestamp'],
                                'clock_target': sat_data.iloc[i]['clock_error'],
                                'ephemeris_target': self._extract_ephemeris_targets(sat_data.iloc[i:i+1])
                            }
                            horizon_targets.append(target)
                
                orbit_targets[horizon] = horizon_targets
            
            target_sequences[orbit_class] = orbit_targets
        
        return target_sequences
    
    def _extract_ephemeris_targets(self, data_subset: pd.DataFrame) -> np.ndarray:
        """Extract ephemeris targets from data subset"""
        
        ephemeris_columns = [col for col in data_subset.columns if col.startswith('ephemeris_')]
        
        if ephemeris_columns:
            ephemeris_data = data_subset[ephemeris_columns].values
            # Ensure we have 6D ephemeris (dx, dy, dz, dvx, dvy, dvz)
            if ephemeris_data.shape[1] >= 6:
                return ephemeris_data[:, :6]
            else:
                # Pad with zeros if insufficient dimensions
                padded = np.zeros((ephemeris_data.shape[0], 6))
                padded[:, :ephemeris_data.shape[1]] = ephemeris_data
                return padded
        else:
            # Return zeros if no ephemeris data
            return np.zeros((len(data_subset), 6))
    
    def _split_train_validation(self, sequences: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict, Dict]:
        """Split sequences into training and validation sets"""
        
        train_data = {}
        val_data = {}
        
        for orbit_class, orbit_sequences in sequences.items():
            n_sequences = len(orbit_sequences)
            n_val = int(n_sequences * self.config.validation_split)
            
            # Random split
            indices = np.random.permutation(n_sequences)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            train_data[orbit_class] = [orbit_sequences[i] for i in train_indices]
            val_data[orbit_class] = [orbit_sequences[i] for i in val_indices]
        
        return train_data, val_data
    
    def _horizon_to_minutes(self, horizon: str) -> int:
        """Convert horizon string to minutes"""
        horizon_map = {
            '15min': 15,
            '30min': 30,
            '1hour': 60,
            '2hour': 120,
            '24hour': 1440
        }
        return horizon_map.get(horizon, 60)

class Day8Predictor:
    """Main Day-8 prediction system"""
    
    def __init__(self, config: Day8PredictionConfig):
        self.config = config
        self.preprocessor = Day8DataPreprocessor(config)
        self.model = None
        self.is_trained = False
        
    def train(self, training_data_points: List[CompetitionDataPoint]) -> Dict[str, Any]:
        """
        Train the Day-8 prediction model
        
        Args:
            training_data_points: 7-day training data
            
        Returns:
            Training metrics and model information
        """
        
        # Prepare training data
        prepared_data = self.preprocessor.prepare_training_data(training_data_points)
        
        # Initialize model
        self.model = CompetitionDualPredictor(self.config.model_config)
        
        # Initialize loss function
        loss_fn = DualPredictorCompetitionLoss(
            horizons=self.config.prediction_horizons,
            normality_weight=self.config.model_config.normality_weight,
            accuracy_weight=self.config.model_config.accuracy_weight
        )
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Training loop
        training_metrics = self._train_model(
            prepared_data['train_data'],
            prepared_data['val_data'],
            loss_fn,
            optimizer
        )
        
        self.is_trained = True
        
        return {
            'training_metrics': training_metrics,
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'config': self.config,
                'data_metadata': prepared_data['metadata']
            }
        }
    
    def predict_day8(self, training_data_points: List[CompetitionDataPoint],
                     day8_data_points: List[CompetitionDataPoint]) -> Dict[str, Any]:
        """
        Predict Day-8 errors using trained model
        
        Args:
            training_data_points: 7-day training data (for context)
            day8_data_points: Day-8 data points to predict
            
        Returns:
            Predictions and evaluation metrics
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        # Prepare Day-8 targets
        day8_targets = self.preprocessor.prepare_day8_targets(day8_data_points)
        
        # Prepare context from training data
        training_context = self.preprocessor.prepare_training_data(training_data_points)
        
        # Make predictions
        predictions = self._make_day8_predictions(training_context, day8_targets)
        
        # Evaluate predictions
        evaluation_metrics = self._evaluate_day8_predictions(predictions, day8_targets)
        
        return {
            'predictions': predictions,
            'evaluation_metrics': evaluation_metrics,
            'day8_metadata': day8_targets['metadata']
        }
    
    def _train_model(self, train_data: Dict, val_data: Dict, 
                    loss_fn: DualPredictorCompetitionLoss,
                    optimizer: torch.optim.Optimizer) -> Dict[str, List[float]]:
        """Train the model with early stopping"""
        
        training_metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_normality': [],
            'val_normality': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.max_epochs):
            # Training phase
            self.model.train()
            train_metrics = self._train_epoch(train_data, loss_fn, optimizer)
            
            # Validation phase
            self.model.eval()
            val_metrics = self._validate_epoch(val_data, loss_fn)
            
            # Record metrics
            training_metrics['train_loss'].append(train_metrics['loss'])
            training_metrics['val_loss'].append(val_metrics['loss'])
            training_metrics['train_normality'].append(train_metrics['normality'])
            training_metrics['val_normality'].append(val_metrics['normality'])
            training_metrics['train_accuracy'].append(train_metrics['accuracy'])
            training_metrics['val_accuracy'].append(val_metrics['accuracy'])
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        return training_metrics
    
    def _train_epoch(self, train_data: Dict, loss_fn: DualPredictorCompetitionLoss,
                    optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Train for one epoch"""
        
        total_loss = 0.0
        total_normality = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for orbit_class, sequences in train_data.items():
            # Create batches
            for i in range(0, len(sequences), self.config.batch_size):
                batch_sequences = sequences[i:i+self.config.batch_size]
                
                if not batch_sequences:
                    continue
                
                # Convert to tensors
                batch_features = torch.stack([torch.tensor(seq['features'], dtype=torch.float32) 
                                            for seq in batch_sequences])
                
                # Create dummy time features (simplified)
                batch_time = torch.randn(len(batch_sequences), self.config.sequence_length, 10)
                
                # Create orbit classes list
                orbit_classes = [seq['orbit_class'] for seq in batch_sequences]
                
                # Forward pass
                predictions = self.model(batch_features, batch_time, orbit_classes)
                
                # Create targets (simplified for training)
                targets = self._create_batch_targets(batch_sequences)
                
                # Compute loss
                loss, metrics = loss_fn(predictions, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Record metrics
                total_loss += loss.item()
                total_normality += metrics.get('ephemeris_metrics', {}).get('normality_score', 0.0)
                total_accuracy += metrics.get('ephemeris_metrics', {}).get('accuracy_score', 0.0)
                num_batches += 1
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'normality': total_normality / max(num_batches, 1),
            'accuracy': total_accuracy / max(num_batches, 1)
        }
    
    def _validate_epoch(self, val_data: Dict, loss_fn: DualPredictorCompetitionLoss) -> Dict[str, float]:
        """Validate for one epoch"""
        
        total_loss = 0.0
        total_normality = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for orbit_class, sequences in val_data.items():
                # Similar to training but without gradient computation
                for i in range(0, len(sequences), self.config.batch_size):
                    batch_sequences = sequences[i:i+self.config.batch_size]
                    
                    if not batch_sequences:
                        continue
                    
                    # Convert to tensors
                    batch_features = torch.stack([torch.tensor(seq['features'], dtype=torch.float32) 
                                                for seq in batch_sequences])
                    batch_time = torch.randn(len(batch_sequences), self.config.sequence_length, 10)
                    orbit_classes = [seq['orbit_class'] for seq in batch_sequences]
                    
                    # Forward pass
                    predictions = self.model(batch_features, batch_time, orbit_classes)
                    targets = self._create_batch_targets(batch_sequences)
                    
                    # Compute loss
                    loss, metrics = loss_fn(predictions, targets)
                    
                    # Record metrics
                    total_loss += loss.item()
                    total_normality += metrics.get('ephemeris_metrics', {}).get('normality_score', 0.0)
                    total_accuracy += metrics.get('ephemeris_metrics', {}).get('accuracy_score', 0.0)
                    num_batches += 1
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'normality': total_normality / max(num_batches, 1),
            'accuracy': total_accuracy / max(num_batches, 1)
        }
    
    def _create_batch_targets(self, batch_sequences: List[Dict[str, Any]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Create target tensors from batch sequences"""
        
        targets = {}
        
        for horizon in self.config.prediction_horizons:
            targets[horizon] = {
                'clock': torch.stack([torch.tensor(seq['clock_targets'][-1:], dtype=torch.float32) 
                                    for seq in batch_sequences]),
                'ephemeris': torch.stack([torch.tensor(seq['ephemeris_targets'][-1:], dtype=torch.float32) 
                                        for seq in batch_sequences])
            }
        
        return targets
    
    def _make_day8_predictions(self, training_context: Dict, day8_targets: Dict) -> Dict[str, Any]:
        """Make predictions for Day-8"""
        
        predictions = {}
        
        self.model.eval()
        with torch.no_grad():
            for orbit_class in day8_targets['target_data'].keys():
                orbit_predictions = {}
                
                for horizon in self.config.prediction_horizons:
                    horizon_targets = day8_targets['target_data'][orbit_class][horizon]
                    
                    # Create dummy features for prediction (simplified)
                    # In practice, this would use the last sequence from training data
                    dummy_features = torch.randn(len(horizon_targets), self.config.sequence_length, 
                                                self.config.model_config.shared_encoder_dim)
                    dummy_time = torch.randn(len(horizon_targets), self.config.sequence_length, 10)
                    orbit_classes = [orbit_class] * len(horizon_targets)
                    
                    # Make predictions
                    horizon_predictions = self.model(dummy_features, dummy_time, orbit_classes)
                    
                    orbit_predictions[horizon] = horizon_predictions[orbit_class][horizon]
                
                predictions[orbit_class] = orbit_predictions
        
        return predictions
    
    def _evaluate_day8_predictions(self, predictions: Dict, day8_targets: Dict) -> CompetitionSubmissionMetrics:
        """Evaluate Day-8 predictions using competition metrics"""
        
        evaluator = CompetitionEvaluator(self.config.prediction_horizons)
        
        # Convert predictions to format expected by evaluator
        formatted_predictions = {'Day8_Model': {}}
        formatted_targets = {}
        
        for orbit_class in predictions.keys():
            for horizon in self.config.prediction_horizons:
                if horizon not in formatted_predictions['Day8_Model']:
                    formatted_predictions['Day8_Model'][horizon] = []
                    formatted_targets[horizon] = []
                
                # Extract predictions and targets
                pred_clock, _ = predictions[orbit_class][horizon]['clock']
                pred_eph, _ = predictions[orbit_class][horizon]['ephemeris']
                
                # Combine clock and ephemeris predictions
                combined_pred = np.concatenate([pred_clock.numpy(), pred_eph.numpy()], axis=-1)
                formatted_predictions['Day8_Model'][horizon].extend(combined_pred)
                
                # Extract targets
                horizon_targets = day8_targets['target_data'][orbit_class][horizon]
                for target in horizon_targets:
                    combined_target = np.concatenate([
                        np.array([target['clock_target']]),
                        target['ephemeris_target'].flatten()
                    ])
                    formatted_targets[horizon].append(combined_target)
        
        # Convert to numpy arrays
        for horizon in formatted_predictions['Day8_Model'].keys():
            formatted_predictions['Day8_Model'][horizon] = np.array(formatted_predictions['Day8_Model'][horizon])
            formatted_targets[horizon] = np.array(formatted_targets[horizon])
        
        # Evaluate
        metrics = evaluator.evaluate_competition_submission(
            formatted_predictions, formatted_targets, model_names=['Day8_Model']
        )
        
        return metrics

# Factory functions
def create_day8_predictor(config: Optional[Day8PredictionConfig] = None) -> Day8Predictor:
    """Create Day-8 predictor with default or custom configuration"""
    if config is None:
        config = Day8PredictionConfig()
    return Day8Predictor(config)

def run_day8_competition(training_data: List[CompetitionDataPoint],
                        day8_data: List[CompetitionDataPoint],
                        config: Optional[Day8PredictionConfig] = None) -> Dict[str, Any]:
    """
    Complete Day-8 competition pipeline
    
    Args:
        training_data: 7-day training data points
        day8_data: Day-8 test data points
        config: Optional configuration
        
    Returns:
        Complete results including predictions, metrics, and model info
    """
    
    # Create predictor
    predictor = create_day8_predictor(config)
    
    # Train model
    training_results = predictor.train(training_data)
    
    # Make Day-8 predictions
    prediction_results = predictor.predict_day8(training_data, day8_data)
    
    # Combine results
    return {
        'training_results': training_results,
        'prediction_results': prediction_results,
        'model': predictor.model,
        'config': predictor.config
    }

# Example usage
if __name__ == "__main__":
    # Create dummy competition data
    import random
    from datetime import datetime, timedelta
    
    # Create 7 days of training data
    training_data = []
    base_time = datetime.now()
    
    for day in range(7):
        for hour in range(0, 24, 0.25):  # 15-minute intervals
            for sat_id in ['SAT001', 'SAT002']:
                orbit_class = 'GEO' if sat_id == 'SAT001' else 'MEO'
                
                dp = CompetitionDataPoint(
                    satellite_id=sat_id,
                    timestamp=base_time + timedelta(days=day, hours=hour),
                    orbit_class=orbit_class,
                    broadcast_clock_bias=random.gauss(0, 1e-6),
                    modeled_clock_bias=random.gauss(0, 1e-6),
                    broadcast_ephemeris={'x': random.gauss(0, 1000), 'y': random.gauss(0, 1000), 'z': random.gauss(0, 1000),
                                       'vx': random.gauss(0, 10), 'vy': random.gauss(0, 10), 'vz': random.gauss(0, 10)},
                    modeled_ephemeris={'x': random.gauss(0, 1000), 'y': random.gauss(0, 1000), 'z': random.gauss(0, 1000),
                                     'vx': random.gauss(0, 10), 'vy': random.gauss(0, 10), 'vz': random.gauss(0, 10)},
                    is_training=True
                )
                dp.compute_errors()
                training_data.append(dp)
    
    # Create Day-8 data
    day8_data = []
    for hour in range(0, 24, 0.25):  # Day 8
        for sat_id in ['SAT001', 'SAT002']:
            orbit_class = 'GEO' if sat_id == 'SAT001' else 'MEO'
            
            dp = CompetitionDataPoint(
                satellite_id=sat_id,
                timestamp=base_time + timedelta(days=7, hours=hour),
                orbit_class=orbit_class,
                broadcast_clock_bias=random.gauss(0, 1e-6),
                modeled_clock_bias=random.gauss(0, 1e-6),
                broadcast_ephemeris={'x': random.gauss(0, 1000), 'y': random.gauss(0, 1000), 'z': random.gauss(0, 1000),
                                   'vx': random.gauss(0, 10), 'vy': random.gauss(0, 10), 'vz': random.gauss(0, 10)},
                modeled_ephemeris={'x': random.gauss(0, 1000), 'y': random.gauss(0, 1000), 'z': random.gauss(0, 1000),
                                 'vx': random.gauss(0, 10), 'vy': random.gauss(0, 10), 'vz': random.gauss(0, 10)},
                is_training=False
            )
            dp.compute_errors()
            day8_data.append(dp)
    
    print(f"Created {len(training_data)} training points and {len(day8_data)} Day-8 test points")
    
    # Run Day-8 competition
    results = run_day8_competition(training_data, day8_data)
    
    print("Day-8 Competition Results:")
    print(f"Final Score: {results['prediction_results']['evaluation_metrics'].final_score:.4f}")
    print(f"Normality Score: {results['prediction_results']['evaluation_metrics'].normality_score:.4f}")
    print(f"Accuracy Score: {results['prediction_results']['evaluation_metrics'].accuracy_score:.4f}")
