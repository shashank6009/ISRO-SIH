import schedule
import time
import subprocess
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from genesis_ai.db.models import TrainingRun, get_engine, get_session, init_database
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('genesis_ai_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GenesisScheduler:
    """Autonomous retraining scheduler for GENESIS-AI."""
    
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.engine = init_database()
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info("GENESIS-AI Scheduler initialized")
    
    def retrain_models(self):
        """Execute nightly model retraining."""
        start_time = datetime.utcnow()
        logger.info(f"[{start_time.isoformat()}] Starting nightly retraining...")
        
        # Look for latest data file
        data_files = list(self.data_dir.glob("*.csv"))
        if not data_files:
            logger.warning("No CSV data files found in data directory")
            return
        
        # Use most recent data file
        latest_data = max(data_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"Using data file: {latest_data}")
        
        # Train both GRU and Transformer models
        models_to_train = ["gru", "transformer"]
        
        for model_type in models_to_train:
            self._train_single_model(model_type, latest_data, start_time)
        
        logger.info("Nightly retraining completed successfully")
    
    def _train_single_model(self, model_type, data_path, start_time):
        """Train a single model and log results."""
        session = get_session(self.engine)
        
        # Create training run record
        training_run = TrainingRun(
            model_type=model_type,
            dataset_path=str(data_path),
            epochs=10,
            batch_size=64,
            learning_rate=1e-3,
            started_at=start_time,
            status="running"
        )
        session.add(training_run)
        session.commit()
        
        try:
            logger.info(f"Training {model_type} model...")
            
            # Build training command
            cmd = [
                sys.executable, "-m", "genesis_ai.training.train",
                "--input_csv", str(data_path),
                "--epochs", "10",
                "--batch_size", "64",
                "--seq_len", "12",
                "--hidden_size", "128",
                "--model_type", model_type,
                "--lr", "1e-3"
            ]
            
            # Execute training
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3600  # 1 hour timeout
            )
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                # Training successful
                training_run.status = "completed"
                training_run.completed_at = end_time
                training_run.training_duration = duration
                logger.info(f"{model_type} training completed successfully in {duration:.1f}s")
            else:
                # Training failed
                training_run.status = "failed"
                training_run.error_message = result.stderr[:1000]  # Truncate error
                logger.error(f"{model_type} training failed: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            training_run.status = "failed"
            training_run.error_message = "Training timeout after 1 hour"
            logger.error(f"{model_type} training timed out")
        
        except Exception as e:
            training_run.status = "failed"
            training_run.error_message = str(e)[:1000]
            logger.error(f"{model_type} training error: {e}")
        
        finally:
            session.commit()
            session.close()
    
    def cleanup_old_records(self, days_to_keep=30):
        """Clean up old database records to prevent unbounded growth."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        session = get_session(self.engine)
        
        try:
            # Clean old forecast records
            deleted_forecasts = session.query(TrainingRun).filter(
                TrainingRun.completed_at < cutoff_date
            ).delete()
            
            session.commit()
            logger.info(f"Cleaned up {deleted_forecasts} old training records")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            session.rollback()
        finally:
            session.close()
    
    def health_check(self):
        """Perform system health check."""
        logger.info("Performing system health check...")
        
        try:
            # Check database connectivity
            session = get_session(self.engine)
            session.execute(text("SELECT 1"))
            session.close()
            logger.info("✅ Database connectivity OK")
            
            # Check data directory
            if self.data_dir.exists():
                data_files = list(self.data_dir.glob("*.csv"))
                logger.info(f"✅ Data directory OK ({len(data_files)} CSV files)")
            else:
                logger.warning("⚠️ Data directory not found")
            
            # Check model directory
            if self.model_dir.exists():
                logger.info("✅ Model directory OK")
            else:
                logger.warning("⚠️ Model directory not found")
                
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
    
    def start_scheduler(self):
        """Start the autonomous scheduler."""
        logger.info("Starting GENESIS-AI autonomous scheduler...")
        
        # Schedule nightly retraining at midnight UTC
        schedule.every().day.at("00:00").do(self.retrain_models)
        
        # Schedule weekly cleanup on Sundays at 1 AM UTC
        schedule.every().sunday.at("01:00").do(self.cleanup_old_records)
        
        # Schedule daily health checks at 6 AM UTC
        schedule.every().day.at("06:00").do(self.health_check)
        
        # Run initial health check
        self.health_check()
        
        logger.info("Scheduler active — retraining every midnight UTC")
        logger.info("Next scheduled jobs:")
        for job in schedule.jobs:
            logger.info(f"  {job}")
        
        # Main scheduler loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            raise

def main():
    """Main entry point for scheduler."""
    scheduler = GenesisScheduler()
    scheduler.start_scheduler()

if __name__ == "__main__":
    main()
