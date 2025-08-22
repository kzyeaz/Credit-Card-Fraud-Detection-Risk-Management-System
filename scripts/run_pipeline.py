"""Main pipeline orchestration script."""

import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion.data_loader import TransactionDataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.fraud_detection import AnomalyDetector
from src.models.balance_forecasting import SimpleForecaster

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )

def run_data_ingestion():
    """Run data ingestion pipeline."""
    print("Running data ingestion...")
    
    loader = TransactionDataLoader()
    loader.create_staging_tables()
    
    # Load transaction data
    if os.path.exists('data/raw/transactions.csv'):
        df = loader.load_raw_transactions('data/raw/transactions.csv')
        quality_metrics = loader.validate_data_quality(df)
        loader.load_to_database(df, 'transactions_raw', if_exists='replace')
        
        # Populate calendar dimension
        min_date = df['timestamp'].min().strftime('%Y-%m-%d')
        max_date = df['timestamp'].max().strftime('%Y-%m-%d')
        loader.populate_calendar_dim(min_date, max_date)
        
        # Load account data
        if os.path.exists('data/raw/accounts.csv'):
            accounts_df = pd.read_csv('data/raw/accounts.csv')
            loader.load_to_database(accounts_df, 'account_dim', if_exists='replace')
        
        print(f"Data ingestion completed: {len(df)} transactions loaded")
        return True
    else:
        print("No transaction data found")
        return False

def run_feature_engineering():
    """Run feature engineering pipeline."""
    print("Running feature engineering...")
    
    # Use simplified feature engineering for large datasets
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from src.features.basic_features import SimpleFeatureEngineer
    fe = SimpleFeatureEngineer()
    features_df = fe.create_features(sample_size=20000)
    
    if len(features_df) > 0:
        # Save to database
        with fe.engine.connect() as conn:
            features_df.to_sql('feature_store', conn, if_exists='replace', index=False)
        print(f"Feature engineering completed: {len(features_df)} records")
        return True
    else:
        print("Feature engineering failed")
        return False

def run_model_training():
    """Run model training pipeline."""
    print("Running model training...")
    
    from sqlalchemy import create_engine
    
    # Load feature data
    engine = create_engine("sqlite:///data/transactions.db")
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM feature_store", conn)
        
        if len(df) == 0:
            print("No feature data available")
            return False
        
        # Train anomaly detection models
        detector = AnomalyDetector(contamination=0.05)
        
        # Get feature columns (exclude non-numeric and ID columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['transaction_id', 'account_id', 'timestamp', 'is_fraud', 'unix_timestamp']
        feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        
        X = detector.prepare_features(df, feature_columns)
        
        detector.train_isolation_forest(X)
        detector.save_models()
        
        print("Anomaly detection models trained")
        
        # Train forecasting models
        forecaster = SimpleForecaster(forecast_horizon=30)
        forecasts_df = forecaster.create_balance_forecasts(sample_size=1000)
        
        if len(forecasts_df) > 0:
            with engine.connect() as conn:
                forecasts_df.to_sql('balance_forecasts', conn, if_exists='replace', index=False)
            
            print(f"Forecasting completed: {len(forecasts_df)} forecasts generated")
        
        return True
        
    except Exception as e:
        print(f"Model training failed: {e}")
        return False

def run_full_pipeline():
    """Run the complete pipeline."""
    print("Starting Customer Transaction Intelligence Pipeline")
    print("=" * 60)
    
    setup_logging()
    
    # Step 1: Data Ingestion
    if not run_data_ingestion():
        print("Pipeline failed at data ingestion")
        return False
    
    # Step 2: Feature Engineering
    if not run_feature_engineering():
        print("Pipeline failed at feature engineering")
        return False
    
    # Step 3: Model Training
    if not run_model_training():
        print("Pipeline failed at model training")
        return False
    
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("\nNext Steps:")
    print("1. Start the dashboard: streamlit run dashboard/app.py")
    print("2. View model performance and alerts")
    print("3. Set up production deployment")
    
    return True

if __name__ == "__main__":
    import pandas as pd
    run_full_pipeline()
