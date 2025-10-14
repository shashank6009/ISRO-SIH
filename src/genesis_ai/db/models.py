from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from pathlib import Path

Base = declarative_base()

class ForecastRecord(Base):
    """Store prediction results for historical analysis and monitoring."""
    __tablename__ = "forecast_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    sat_id = Column(String(50), nullable=False)
    orbit_class = Column(String(20), nullable=False)
    quantity = Column(String(20), nullable=False)  # 'clock' or 'ephem'
    forecast_timestamp = Column(DateTime, nullable=False)  # When prediction was made
    target_timestamp = Column(DateTime, nullable=False)    # Target time being predicted
    predicted_error = Column(Float, nullable=False)
    lower_bound = Column(Float)
    upper_bound = Column(Float)
    model_type = Column(String(50), nullable=False)
    seq_len = Column(Integer)
    hidden_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class TrainingRun(Base):
    """Track model training sessions and performance."""
    __tablename__ = "training_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False)
    dataset_path = Column(String(500))
    epochs = Column(Integer)
    batch_size = Column(Integer)
    learning_rate = Column(Float)
    final_loss = Column(Float)
    validation_loss = Column(Float)
    training_duration = Column(Float)  # seconds
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    status = Column(String(20), default="running")  # running, completed, failed
    error_message = Column(String(1000))

class AlertRecord(Base):
    """Log anomaly alerts and system notifications."""
    __tablename__ = "alert_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String(50), nullable=False)  # anomaly, system_error, etc.
    severity = Column(String(20), nullable=False)    # low, medium, high, critical
    sat_id = Column(String(50))
    message = Column(String(1000), nullable=False)
    details = Column(String(5000))  # JSON or detailed info
    acknowledged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))

def get_engine(db_url=None):
    """Create database engine with fallback to SQLite."""
    if db_url is None:
        # Default to SQLite in project directory
        db_path = Path(__file__).parent.parent.parent.parent / "genesis_ai.db"
        db_url = f"sqlite:///{db_path}"
    
    # Handle PostgreSQL vs SQLite
    if db_url.startswith("postgresql"):
        return create_engine(db_url, echo=False, future=True, pool_pre_ping=True)
    else:
        return create_engine(db_url, echo=False, future=True)

def get_session(engine):
    """Create database session."""
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    return Session()

def init_database(db_url=None):
    """Initialize database tables."""
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return engine

def get_db_stats(engine):
    """Get database statistics for monitoring."""
    session = get_session(engine)
    try:
        stats = {
            "forecast_records": session.query(ForecastRecord).count(),
            "training_runs": session.query(TrainingRun).count(),
            "alert_records": session.query(AlertRecord).count(),
            "latest_forecast": session.query(ForecastRecord.created_at).order_by(ForecastRecord.created_at.desc()).first(),
            "latest_training": session.query(TrainingRun.completed_at).order_by(TrainingRun.completed_at.desc()).first(),
        }
        return stats
    finally:
        session.close()

if __name__ == "__main__":
    # Initialize database when run directly
    print("Initializing GENESIS-AI database...")
    engine = init_database()
    print("Database initialized successfully!")
    
    # Show stats
    stats = get_db_stats(engine)
    print("Database statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
