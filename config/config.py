"""Configuration settings for the Customer Transaction Intelligence system."""

import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    
    # Anomaly Detection
    isolation_forest_contamination: float = 0.1
    isolation_forest_n_estimators: int = 100
    lof_n_neighbors: int = 20
    autoencoder_encoding_dim: int = 32
    autoencoder_epochs: int = 100
    
    # Forecasting
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_prior_scale: float = 10.0
    prophet_forecast_horizon: int = 30
    sarimax_order: tuple = (1, 1, 1)
    sarimax_seasonal_order: tuple = (1, 1, 1, 7)
    
    # Feature Engineering
    rolling_windows: List[int] = None
    min_transactions_for_training: int = 50
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [1, 7, 14, 30]

@dataclass
class DataConfig:
    """Data configuration parameters."""
    
    # Paths
    raw_data_path: str = "data/raw/"
    processed_data_path: str = "data/processed/"
    models_path: str = "models/"
    
    # Database
    db_connection_string: str = "sqlite:///data/transactions.db"
    
    # Data Quality
    max_missing_rate: float = 0.1
    duplicate_threshold: float = 0.05

@dataclass
class AlertConfig:
    """Alert configuration parameters."""
    
    # Thresholds
    high_risk_threshold: float = 0.8
    medium_risk_threshold: float = 0.6
    overdraft_risk_days: int = 7
    
    # Alert Management
    max_alerts_per_day: int = 100
    alert_cooldown_hours: int = 24
    auto_close_threshold: float = 0.3

@dataclass
class SystemConfig:
    """System configuration parameters."""
    
    # Processing
    batch_size: int = 1000
    max_workers: int = 4
    
    # Monitoring
    model_retrain_days: int = 30
    performance_check_days: int = 7
    
    # API
    api_timeout: int = 30
    max_requests_per_minute: int = 100

# Global configuration instance
config = {
    'model': ModelConfig(),
    'data': DataConfig(),
    'alert': AlertConfig(),
    'system': SystemConfig()
}

# Environment variables
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')
DATABASE_URL = os.getenv('DATABASE_URL', config['data'].db_connection_string)
